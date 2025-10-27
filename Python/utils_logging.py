import json
import logging
import logging.handlers
import time
import traceback
from datetime import datetime, timezone
from uuid import uuid4

class JSONLineFormatter(logging.Formatter):
    def __init__(self, static_fields=None):
        super().__init__()
        self.static_fields = static_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        # Start with standard fields
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Merge extra fields (everything in record.__dict__ that looks custom)
        for k, v in record.__dict__.items():
            if k in ("msg", "args", "levelname", "levelno", "pathname", "filename",
                     "module", "exc_info", "exc_text", "stack_info", "lineno",
                     "funcName", "created", "msecs", "relativeCreated", "thread",
                     "threadName", "processName", "process"):
                continue
            payload[k] = v
        # Attach static (run-wide) fields last so they’re always present
        payload.update(self.static_fields)
        # Exception info if present
        if record.exc_info:
            payload["exc_type"] = record.exc_info[0].__name__
            payload["exc"] = "".join(traceback.format_exception(*record.exc_info)).rstrip()
        return json.dumps(payload, ensure_ascii=False)

def build_logger(name: str, json_path: str, static_fields: dict) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # avoid duplicate logs if root has handlers

    fmt = JSONLineFormatter(static_fields=static_fields)

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Rotating file (keep last 5 files of 10 MB each)
    fh = logging.handlers.RotatingFileHandler(
        json_path, maxBytes=10_000_000, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger

# ---- Example atmospheric correction pipeline ----

def rayleigh_correction(image_path, sensor: str, altitude_km: float):
    # dummy placeholder computation
    time.sleep(0.05)
    return {"status": "ok", "path": image_path.replace(".tif", "_rayleigh.tif")}

def aerosol_correction(image_path, model: str, vis_km: float):
    time.sleep(0.1)
    if vis_km <= 0:
        raise ValueError("Visibility (km) must be > 0")
    return {"status": "ok", "path": image_path.replace(".tif", "_aerosol.tif")}

def atmospheric_correction(scene_id: str, image_path: str, logger: logging.Logger, params: dict):
    t0 = time.perf_counter()
    logger.info("start atmospheric correction",
                extra={"event": "start", "scene_id": scene_id, "image_path": image_path, "params": params})
    try:
        t = time.perf_counter()
        out1 = rayleigh_correction(image_path, params["sensor"], params["target_alt_km"])
        logger.info("rayleigh correction done",
                    extra={"event": "step_done", "step": "rayleigh",
                           "duration_s": round(time.perf_counter() - t, 4),
                           "output": out1})

        t = time.perf_counter()
        out2 = aerosol_correction(out1["path"], params["aerosol_model"], params["visibility_km"])
        logger.info("aerosol correction done",
                    extra={"event": "step_done", "step": "aerosol",
                           "duration_s": round(time.perf_counter() - t, 4),
                           "output": out2})

        total = round(time.perf_counter() - t0, 4)
        logger.info("atmospheric correction completed",
                    extra={"event": "end", "scene_id": scene_id, "total_duration_s": total,
                           "final_output": out2["path"]})
        return out2["path"]

    except Exception:
        # log with full traceback in JSON
        logger.exception("pipeline failed",
                         extra={"event": "error", "scene_id": scene_id})
        raise

if __name__ == "__main__":
    run_id = str(uuid4())
    static = {
        "run_id": run_id,
        "pipeline": "atmo_corr_v1",
        "algorithm": "ACME-Atmos-2025.09",
        "host": "worker-03",  # inject your hostname/container id
    }
    logger = build_logger("atmo", "atmo_correction.jsonl", static)

    # Example parameters you might load from a config file
    params = {
        "sensor": "S2A",
        "aerosol_model": "continental",
        "visibility_km": 35.0,
        "target_alt_km": 2.0,
        "sun_zenith_deg": 24.7,
        "atm_profile": "midlatitude_summer",
    }

    scene_id = "S2A_MSIL1C_20250901T103021_N0510_R108_T31TCJ_20250901T124512"
    output = atmospheric_correction(scene_id, "/data/scene.tif", logger, params)
    print("Final product:", output)
