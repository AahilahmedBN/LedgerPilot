import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from generate_test_documents import generate_all, SAMPLE_RECEIPTS
from preprocessing import preprocess_pipeline
from ocr_extract import extract_text


def run():
    print("=" * 70)
    print("STEP 1: Generating synthetic test documents (stopgap for real photos)")
    print("=" * 70)
    doc_paths = generate_all(out_dir=os.path.join(os.path.dirname(__file__), "sample_docs"))

    print(f"\n{'=' * 70}")
    print("STEP 2: Running preprocessing + OCR on each document")
    print("=" * 70)

    results = []
    for path in doc_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        intermediate_dir = os.path.join(os.path.dirname(__file__), "processed_output", name)

        print(f"\n--- {name} ---")
        processed_img, meta = preprocess_pipeline(path, save_intermediate_dir=intermediate_dir)
        print(f"  Auto-cropped: {meta['was_cropped']} | Deskew angle: {meta['deskew_angle_deg']} deg")

        text, confidence = extract_text(processed_img)
        print(f"  Mean OCR confidence: {confidence}")
        print(f"  Extracted text:\n{'-' * 40}")
        print("  " + text.replace("\n", "\n  "))
        print("-" * 40)

        results.append({
            "document": name,
            "was_cropped": meta["was_cropped"],
            "deskew_angle": meta["deskew_angle_deg"],
            "mean_confidence": confidence,
            "extracted_text": text,
        })

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    avg_conf = sum(r["mean_confidence"] for r in results) / len(results)
    print(f"Documents processed: {len(results)}")
    print(f"Average OCR confidence: {round(avg_conf, 1)}")
    print(f"Intermediate stage images saved under: processed_output/<doc_name>/")
    print("\nHonest note for your report: confidence numbers here are against")
    print("SYNTHETIC degraded images, not real phone photos. Expect real-photo")
    print("confidence to be lower until you validate against actual documents —")
    print("report both numbers separately once you have real data, don't")
    print("conflate them.")

    return results


if __name__ == "__main__":
    run()
