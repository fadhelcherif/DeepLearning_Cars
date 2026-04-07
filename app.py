import os
import json
import csv
import time
import re
from difflib import SequenceMatcher
from io import BytesIO
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import parse_qs, urlparse

import requests
import torch
import timm
from flask import Flask, render_template, request
from PIL import Image
from torchvision import transforms
from torchvision import models as tv_models
from torchvision.transforms import InterpolationMode
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "car_model.pth"
MODEL_URL = os.getenv("MODEL_URL", "").strip()
MODEL_DOWNLOAD_TIMEOUT_SECONDS = int(os.getenv("MODEL_DOWNLOAD_TIMEOUT_SECONDS", "180"))
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
IMG_SIZE = 300
DEFAULT_CURRENCY = "DT"
SCRAPED_DATA_PATH = Path(os.getenv("SCRAPED_DATA_PATH", str(BASE_DIR / "cars_data.json")))
SCRAPED_DATA_URL = os.getenv("SCRAPED_DATA_URL", "").strip()
GOOGLE_SHEETS_CSV_URL = os.getenv(
    "GOOGLE_SHEETS_CSV_URL",
    "https://docs.google.com/spreadsheets/d/1xSv2tbqVddoh2ID78onbnjRjRQq46ThOsDEeHUc2a6I/edit?usp=sharing",
).strip()
SHEETS_REFRESH_SECONDS = int(os.getenv("SHEETS_REFRESH_SECONDS", "300"))
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "").strip()
N8N_TIMEOUT_SECONDS = int(os.getenv("N8N_TIMEOUT_SECONDS", "15"))

_SHEETS_CACHE_ITEMS: List[Dict[str, Any]] = []
_SHEETS_CACHE_AT: float = 0.0


def _normalize_model_url(url: str) -> str:
    parsed = urlparse(url)
    if "drive.google.com" not in parsed.netloc.lower():
        return url

    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) >= 3 and path_parts[0] == "file" and path_parts[1] == "d":
        file_id = path_parts[2]
        return f"https://drive.google.com/uc?export=download&id={file_id}"

    query = parse_qs(parsed.query)
    if "id" in query and query["id"]:
        return f"https://drive.google.com/uc?export=download&id={query['id'][0]}"

    return url


def ensure_model_available() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH

    if not MODEL_URL:
        raise FileNotFoundError(
            f"Model file was not found: {MODEL_PATH}. Set MODEL_URL to a direct download link for the checkpoint."
        )

    model_url = _normalize_model_url(MODEL_URL)
    temp_path = MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".download")

    try:
        print(f"Model missing locally. Downloading from MODEL_URL to {MODEL_PATH} ...")
        response = requests.get(
            model_url,
            stream=True,
            timeout=MODEL_DOWNLOAD_TIMEOUT_SECONDS,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()

        bytes_written = 0
        with temp_path.open("wb") as file_obj:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                file_obj.write(chunk)
                bytes_written += len(chunk)

        if bytes_written == 0:
            raise RuntimeError("Downloaded model is empty. Check MODEL_URL permissions/link type.")

        temp_path.replace(MODEL_PATH)
        print(f"Model downloaded successfully ({bytes_written} bytes).")
        return MODEL_PATH
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def build_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(int(IMG_SIZE * 1.12), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def infer_num_classes(state_dict: Dict[str, torch.Tensor], default: int = 196) -> int:
    for head_key in ("classifier.weight", "fc.weight"):
        if head_key in state_dict and hasattr(state_dict[head_key], "shape"):
            return int(state_dict[head_key].shape[0])
    return default


def looks_like_efficientnet(keys: List[str]) -> bool:
    return any(k.startswith("conv_stem") or k.startswith("blocks.") for k in keys)


def looks_like_resnet(keys: List[str]) -> bool:
    return any(k.startswith("layer1.") or k.startswith("fc.") or k.startswith("conv1.") for k in keys)


def pick_resnet_variant(keys: List[str]) -> str:
    if any(".conv3." in k for k in keys):
        return "resnet50"

    max_layer3_block = -1
    for k in keys:
        if k.startswith("layer3."):
            parts = k.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                max_layer3_block = max(max_layer3_block, int(parts[1]))

    return "resnet34" if max_layer3_block >= 3 else "resnet18"


def load_label_mapping(num_classes: int) -> Dict[int, str]:
    # Try to fetch Stanford Cars label names. If unavailable, use generic labels.
    try:
        from datasets import load_dataset_builder

        builder = load_dataset_builder("tanganke/stanford_cars")
        names = builder.info.features["label"].names
        if len(names) == num_classes:
            return {i: n for i, n in enumerate(names)}
    except Exception:
        pass

    return {i: f"class_{i}" for i in range(num_classes)}


def load_model_and_labels() -> Tuple[torch.nn.Module, torch.device, transforms.Compose, Dict[int, str]]:
    model_path = ensure_model_available()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    # Remove DataParallel prefix if present.
    state_dict = {
        (k.replace("module.", "", 1) if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }
    keys = list(state_dict.keys())

    num_classes = infer_num_classes(state_dict)
    if looks_like_efficientnet(keys):
        model = timm.create_model("efficientnetv2_rw_m", pretrained=False, num_classes=num_classes)
    elif looks_like_resnet(keys):
        variant = pick_resnet_variant(keys)
        model = getattr(tv_models, variant)(weights=None, num_classes=num_classes)
    else:
        raise RuntimeError("Could not infer model architecture from checkpoint keys.")

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    idx_to_class = load_label_mapping(num_classes)
    eval_transform = build_eval_transform()
    return model, device, eval_transform, idx_to_class


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_topk(
    model: torch.nn.Module,
    image: Image.Image,
    image_transform: transforms.Compose,
    idx_to_class: Dict[int, str],
    device: torch.device,
    k: int = 3,
) -> List[Tuple[str, float]]:
    if image.mode != "RGB":
        image = image.convert("RGB")

    with torch.inference_mode():
        x = image_transform(image).unsqueeze(0).to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        top_prob, top_cls = torch.topk(probs, k=k)

    results: List[Tuple[str, float]] = []
    for rank in range(k):
        cls_id = int(top_cls[0][rank].item())
        confidence = float(top_prob[0][rank].item())
        label = idx_to_class.get(cls_id, f"class_{cls_id}")
        results.append((label, confidence))

    return results


def _normalize_label(value: str) -> str:
    return " ".join(value.lower().strip().split())


def _normalize_for_match(value: str) -> str:
    cleaned = _normalize_label(value)
    cleaned = re.sub(r"\b(19|20)\d{2}\b", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", cleaned)
    return " ".join(cleaned.split())


def _token_set(value: str) -> set:
    stop_words = {
        "sedan",
        "hatchback",
        "coupe",
        "convertible",
        "cabriolet",
        "wagon",
        "suv",
        "van",
    }
    return {token for token in _normalize_for_match(value).split() if token and token not in stop_words}


def _extract_items_from_json(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if isinstance(payload, dict):
        for key in ("cars", "prices", "results", "data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]

    return []


def _extract_label(item: Dict[str, Any]) -> str:
    for key in ("label", "name", "model", "title", "car_name"):
        value = item.get(key)
        if value:
            return str(value)
    return ""


def _extract_price(item: Dict[str, Any]) -> Any:
    for key in ("price", "avg_price", "current_price", "amount"):
        if key in item:
            return item.get(key)
    return None


def _coerce_price_number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)

    if value is None:
        return None

    text = str(value).strip().replace("\u00a0", " ")
    if not text:
        return None

    cleaned = re.sub(r"[^0-9,.-]", "", text)
    if not cleaned:
        return None

    if "," in cleaned and "." in cleaned:
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned and "." not in cleaned:
        cleaned = cleaned.replace(",", "")

    try:
        return float(cleaned)
    except ValueError:
        return None


def _format_average_price(value: float) -> Any:
    rounded_int = round(value)
    if abs(value - rounded_int) < 1e-9:
        return int(rounded_int)
    return round(value, 2)


def _normalize_data_source_url(url: str) -> str:
    parsed = urlparse(url)
    if "drive.google.com" not in parsed.netloc.lower():
        return url

    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) >= 3 and path_parts[0] == "file" and path_parts[1] == "d":
        file_id = path_parts[2]
        return f"https://drive.google.com/uc?export=download&id={file_id}"

    query_values = parse_qs(parsed.query)
    file_id_values = query_values.get("id", [])
    if file_id_values:
        file_id = file_id_values[0]
        return f"https://drive.google.com/uc?export=download&id={file_id}"

    return url


def _normalize_google_sheets_csv_url(url: str) -> str:
    parsed = urlparse(url)
    if "docs.google.com" not in parsed.netloc.lower() or "/spreadsheets/" not in parsed.path:
        return url

    path_parts = [part for part in parsed.path.split("/") if part]
    sheet_id = ""
    if len(path_parts) >= 3 and path_parts[0] == "spreadsheets" and path_parts[1] == "d":
        sheet_id = path_parts[2]

    if not sheet_id:
        return url

    query_values = parse_qs(parsed.query)
    gid_values = query_values.get("gid", [])
    gid = gid_values[0] if gid_values else "0"

    if parsed.fragment and "gid=" in parsed.fragment:
        fragment_qs = parse_qs(parsed.fragment)
        gid = fragment_qs.get("gid", [gid])[0]

    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"


def _load_items_from_google_sheets() -> Tuple[List[Dict[str, Any]], str]:
    global _SHEETS_CACHE_ITEMS, _SHEETS_CACHE_AT

    if not GOOGLE_SHEETS_CSV_URL:
        return [], "GOOGLE_SHEETS_CSV_URL is empty."

    now = time.time()
    if _SHEETS_CACHE_ITEMS and (now - _SHEETS_CACHE_AT) < SHEETS_REFRESH_SECONDS:
        return _SHEETS_CACHE_ITEMS, ""

    resolved_url = _normalize_google_sheets_csv_url(GOOGLE_SHEETS_CSV_URL)

    try:
        response = requests.get(
            resolved_url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
    except Exception as exc:
        return [], f"Could not read Google Sheets CSV: {exc}"

    try:
        reader = csv.DictReader(StringIO(response.text))
        items = [dict(row) for row in reader if isinstance(row, dict)]
    except Exception as exc:
        return [], f"Could not parse Google Sheets CSV: {exc}"

    if not items:
        return [], "Google Sheets CSV loaded, but no rows were found."

    _SHEETS_CACHE_ITEMS = items
    _SHEETS_CACHE_AT = now
    return items, ""


def _load_scraped_payload() -> Tuple[Any, str]:
    if SCRAPED_DATA_URL:
        resolved_url = _normalize_data_source_url(SCRAPED_DATA_URL)
        try:
            response = requests.get(
                resolved_url,
                timeout=15,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()
            return response.json(), ""
        except Exception as exc:
            return None, f"Could not read JSON from SCRAPED_DATA_URL: {exc}"

    if not SCRAPED_DATA_PATH.exists():
        return None, f"Scraped data file not found at {SCRAPED_DATA_PATH}."

    try:
        with SCRAPED_DATA_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh), ""
    except Exception as exc:
        return None, f"Could not read scraped JSON from file: {exc}"


def _best_match_price(label: str, price_map: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    normalized_label = _normalize_label(label)
    if normalized_label in price_map:
        return price_map[normalized_label]

    # Fallback matching for scraped labels that are not exact model names.
    for key, value in price_map.items():
        if normalized_label in key or key in normalized_label:
            return value

    label_match = _normalize_for_match(label)
    label_tokens = _token_set(label)
    best_score = 0.0
    best_value: Dict[str, object] = {}

    for key, value in price_map.items():
        key_match = _normalize_for_match(key)
        key_tokens = _token_set(key)

        if not key_match:
            continue

        sequence_score = SequenceMatcher(None, label_match, key_match).ratio()

        token_score = 0.0
        if label_tokens and key_tokens:
            intersection = len(label_tokens.intersection(key_tokens))
            union = len(label_tokens.union(key_tokens))
            token_score = intersection / union if union else 0.0

        score = max(sequence_score, token_score)
        if score > best_score:
            best_score = score
            best_value = value

    # Keep threshold conservative to avoid wrong prices on unrelated models.
    if best_score >= 0.45:
        return best_value

    return {}


def _build_price_lookup(items: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, object]], List[Dict[str, object]]]:
    price_map: Dict[str, Dict[str, object]] = {}
    ordered_prices: List[Dict[str, object]] = []
    grouped: Dict[str, Dict[str, Any]] = {}

    for item in items:
        label = _extract_label(item).strip()
        price = _extract_price(item)
        currency = item.get("currency", DEFAULT_CURRENCY)

        ordered_prices.append({"price": price, "currency": currency})

        if label:
            key = _normalize_label(label)
            entry = grouped.setdefault(
                key,
                {
                    "sum": 0.0,
                    "count": 0,
                    "last_price": price,
                    "currency": currency,
                },
            )

            numeric_price = _coerce_price_number(price)
            if numeric_price is not None:
                entry["sum"] += numeric_price
                entry["count"] += 1
            entry["last_price"] = price
            if currency:
                entry["currency"] = currency

    for key, entry in grouped.items():
        if entry["count"] > 0:
            avg_price = _format_average_price(entry["sum"] / entry["count"])
        else:
            avg_price = entry["last_price"]

        price_map[key] = {
            "price": avg_price,
            "currency": entry["currency"],
        }

    return price_map, ordered_prices


def fetch_prices_from_n8n(predicted_labels: List[str]) -> Tuple[Dict[str, Dict[str, object]], List[Dict[str, object]], str]:
    if not predicted_labels:
        return {}, [], ""

    if not N8N_WEBHOOK_URL:
        return {}, [], "N8N_WEBHOOK_URL is empty."

    payload: Any = None
    errors: List[str] = []
    candidate_urls = [N8N_WEBHOOK_URL]
    if "/webhook/" in N8N_WEBHOOK_URL:
        candidate_urls.append(N8N_WEBHOOK_URL.replace("/webhook/", "/webhook-test/", 1))
    elif "/webhook-test/" in N8N_WEBHOOK_URL:
        candidate_urls.append(N8N_WEBHOOK_URL.replace("/webhook-test/", "/webhook/", 1))

    for url in candidate_urls:
        try:
            response = requests.post(
                url,
                json={"predictions": predicted_labels},
                timeout=N8N_TIMEOUT_SECONDS,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()
            payload = response.json()
            break
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    if payload is None:
        return {}, [], "Could not fetch price data from n8n. " + " | ".join(errors[:2])

    items = _extract_items_from_json(payload)
    if not items:
        return {}, [], "n8n returned JSON, but no car items were found."

    price_map, ordered_prices = _build_price_lookup(items)
    return price_map, ordered_prices, ""


def fetch_prices_from_json(predicted_labels: List[str]) -> Tuple[Dict[str, Dict[str, object]], List[Dict[str, object]], str]:
    if not predicted_labels:
        return {}, [], ""

    payload, load_error = _load_scraped_payload()
    if load_error:
        return {}, [], load_error

    items = _extract_items_from_json(payload)
    if not items:
        return {}, [], "JSON file loaded, but no car items were found."
    price_map, ordered_prices = _build_price_lookup(items)
    return price_map, ordered_prices, ""


def fetch_prices_from_google_sheets(predicted_labels: List[str]) -> Tuple[Dict[str, Dict[str, object]], List[Dict[str, object]], str]:
    if not predicted_labels:
        return {}, [], ""

    items, sheets_error = _load_items_from_google_sheets()
    if sheets_error:
        return {}, [], sheets_error

    price_map, ordered_prices = _build_price_lookup(items)
    return price_map, ordered_prices, ""


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
model, device, eval_transform, idx_to_class = load_model_and_labels()


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    results = None
    image_url = None
    integration_warning = None

    if request.method == "POST":
        try:
            image_link = request.form.get("image_link", "").strip()
            file = request.files.get("image")

            if image_link:
                response = requests.get(
                    image_link,
                    timeout=12,
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                image_url = image_link
            else:
                if not file or file.filename == "":
                    error = "Please upload an image or paste an image link."
                    return render_template("index.html", error=error, results=results, image_url=image_url)

                if not allowed_file(file.filename):
                    error = "Unsupported file type. Use png, jpg, jpeg, or webp."
                    return render_template("index.html", error=error, results=results, image_url=image_url)

                safe_name = secure_filename(file.filename)
                save_path = UPLOAD_DIR / safe_name
                file.save(save_path)
                image = Image.open(save_path)
                image_url = f"/static/uploads/{safe_name}"

            raw_results = predict_topk(model, image, eval_transform, idx_to_class, device, k=3)
            labels = [label for label, _ in raw_results]

            price_map, ordered_prices, price_warning = fetch_prices_from_google_sheets(labels)
            if price_warning:
                json_price_map, json_ordered_prices, json_warning = fetch_prices_from_json(labels)
                if json_price_map or json_ordered_prices:
                    price_map, ordered_prices = json_price_map, json_ordered_prices
                    price_warning = (
                        "Google Sheets failed, using JSON fallback. "
                        + price_warning
                    )
                elif json_warning:
                    price_warning = price_warning + " JSON fallback also failed: " + json_warning

            if price_warning:
                integration_warning = (
                    "Price data warning: "
                    + price_warning
                    + " Configure GOOGLE_SHEETS_CSV_URL or use SCRAPED_DATA_URL/SCRAPED_DATA_PATH."
                )

            results = []
            for i, (label, _) in enumerate(raw_results, start=1):
                price_info = _best_match_price(label, price_map)

                results.append(
                    {
                        "pick": i,
                        "label": label,
                        "price": price_info.get("price"),
                        "currency": price_info.get("currency", DEFAULT_CURRENCY),
                    }
                )
        except Exception as exc:
            error = f"Prediction failed: {exc}"

    return render_template(
        "index.html",
        error=error,
        results=results,
        image_url=image_url,
        integration_warning=integration_warning,
    )


if __name__ == "__main__":
    app.run()
