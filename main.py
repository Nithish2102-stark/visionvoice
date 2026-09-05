#!/usr/bin/env python3
"""
VisionVoice - Assistive Reading Device Main Entry Point.
Supports macOS development and Raspberry Pi 4 deployment.
"""

from __future__ import annotations
import argparse
import sys
import os
import time
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from visionvoice.utils.config import get_config
from visionvoice.utils.logging import get_logger, setup_logging
from visionvoice.utils.text import split_into_sentences, detect_primary_language, LANGUAGE_NAMES


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VisionVoice Assistive Reading Device",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                     # Run primary voice-controlled assistive reading flow
  python main.py --dev               # Run in development mode with camera preview HUD
  python main.py --camera-test       # Test camera capture and resolution
  python main.py --ocr sample.jpg    # Run advanced OCR pipeline on a static image
  python main.py --test-tts          # Test multilingual TTS audio playback
  python main.py --test-voice        # Test microphone & voice command recognition
        """
    )

    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enable Development Mode (live camera preview window, keyboard shortcuts)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run an automated end-to-end simulated demonstration on Mac",
    )
    parser.add_argument(
        "--camera-test",
        action="store_true",
        help="Launch interactive camera test feed",
    )
    parser.add_argument(
        "--ocr",
        type=str,
        metavar="IMAGE_PATH",
        help="Run full multi-variant OCR engine on a static book page image and print diagnostic scores",
    )
    parser.add_argument(
        "--test-tts",
        action="store_true",
        help="Test Text-to-Speech playback across supported languages",
    )
    parser.add_argument(
        "--test-voice",
        action="store_true",
        help="Test live microphone voice recognition and command intent parsing",
    )
    parser.add_argument(
        "--test-page",
        type=str,
        metavar="IMAGE_PATH",
        help="Test page detection and perspective dewarping on an image",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run OCR in fast mode (evaluates top 4 variants)",
    )

    return parser.parse_args()


def run_camera_test() -> None:
    """Tests camera hardware and displays live stream."""
    logger = get_logger("CameraTest")
    from visionvoice.camera.base import get_camera
    
    logger.info("Initializing camera for test...")
    cam = get_camera()
    if not cam.start():
        logger.error("Failed to start camera device.")
        return

    logger.info("Camera stream active. Press 'q' or ESC in preview window to exit. Press 's' to capture a test frame.")
    cv2.namedWindow("VisionVoice Camera Test", cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = cam.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            h, w = frame.shape[:2]
            cv2.putText(
                frame,
                f"Resolution: {w}x{h} | Press 's' to capture, 'q' to quit",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("VisionVoice Camera Test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                cfg = get_config()
                test_path = cfg.CAPTURES_DIR / "camera_test_capture.jpg"
                cv2.imwrite(str(test_path), frame)
                logger.info(f"Saved test capture to: {test_path}")

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        logger.info("Camera test finished.")


def run_ocr_test(image_path: str, fast_mode: bool = False) -> None:
    """Runs the advanced OCR pipeline on a given image file and displays diagnostics."""
    logger = get_logger("OCRTest")
    from visionvoice.ocr.engine import OCREngine

    path = Path(image_path).resolve()
    if not path.exists():
        logger.error(f"Image path does not exist: {path}")
        return

    logger.info(f"Loading image from: {path}")
    image = cv2.imread(str(path))
    if image is None:
        logger.error("Failed to decode image file.")
        return

    logger.info(f"Image dimensions: {image.shape[1]}x{image.shape[0]}")
    engine = OCREngine()

    logger.info("Starting High-Accuracy Multi-Variant OCR Pipeline...")
    result = engine.process_image(image, fast_mode=fast_mode)

    print("\n" + "=" * 60)
    print("           VISIONVOICE OCR EVALUATION REPORT          ")
    print("=" * 60)
    print(f"Status              : {'SUCCESS' if result.is_valid else 'FAILED'}")
    if result.error_message:
        print(f"Error               : {result.error_message}")
    print(f"Selected Variant    : {result.selected_variant}")
    print(f"Selected PSM        : {result.selected_psm}")
    print(f"Composite Score     : {result.composite_score:.2f} / 100")
    print(f"Average Confidence  : {result.average_confidence:.1f}%")
    print(f"Detected Language   : {result.detected_script} ({result.detected_language})")
    print(f"Sentence Count      : {len(result.sentences)}")
    print(f"Total Characters    : {len(result.cleaned_text)}")
    print(f"Original Image Saved: {result.original_image_path}")
    print(f"Processed Image     : {result.processed_image_path}")
    print("-" * 60)
    print("CLEANED OCR TEXT:")
    print("-" * 60)
    print(result.cleaned_text if result.cleaned_text else "[No text detected]")
    print("-" * 60)
    print("SEGMENTED SENTENCES FOR TTS:")
    print("-" * 60)
    for i, s in enumerate(result.sentences, 1):
        print(f"[{i:02d}] {s}")
    print("=" * 60 + "\n")


def run_tts_test() -> None:
    """Tests TTS engine across languages."""
    logger = get_logger("TTSTest")
    from visionvoice.speech.base import get_tts

    tts = get_tts()
    samples = [
        ("en", "Hello! VisionVoice is ready to assist you with reading."),
        ("hi", "नमस्ते! विज़नवॉइस आपकी पुस्तक पढ़ने के लिए तैयार है।"),
        ("ta", "வணக்கம்! விஷன் வாய்ஸ் புத்தகங்களை வாசிக்க தயாராக உள்ளது."),
    ]

    logger.info("Starting Multilingual TTS Test...")
    for lang, text in samples:
        logger.info(f"Testing TTS for [{lang}]: '{text}'")
        tts.speak(text, language=lang, blocking=True)
        time.sleep(0.5)

    logger.info("TTS Test completed.")


def run_voice_test() -> None:
    """Tests voice command listener."""
    logger = get_logger("VoiceTest")
    from visionvoice.voice.controller import VoiceController

    vc = VoiceController()
    logger.info("==================================================")
    logger.info(" VOICE COMMAND RECOGNITION TEST                   ")
    logger.info(" Say commands like:                               ")
    logger.info(" - 'Hey VisionVoice'                              ")
    logger.info(" - 'Start reading'                                ")
    logger.info(" - 'Pause' / 'Resume' / 'Repeat'                  ")
    logger.info(" - 'Stop reading' (to exit test)                  ")
    logger.info("==================================================")

    if not vc.has_microphone:
        logger.warning("No microphone detected. Enter commands via console or run with PyAudio.")
        return

    while True:
        logger.info("Listening for command...")
        intent, arg = vc.listen_single_utterance(timeout=7.0)
        logger.info(f"--> Recognized Intent: {intent.value} | Arg: {arg}")
        if intent.value == "STOP_READING":
            logger.info("Stop command received. Exiting test.")
            break


def run_page_detection_test(image_path: str) -> None:
    """Tests page quadrilateral detection and perspective transform on a static image."""
    logger = get_logger("PageDetectionTest")
    from visionvoice.ocr.page_detector import PageDetector

    path = Path(image_path).resolve()
    if not path.exists():
        logger.error(f"Image not found: {path}")
        return

    image = cv2.imread(str(path))
    if image is None:
        logger.error("Failed to load image.")
        return

    detector = PageDetector()
    analysis = detector.analyze_frame(image, draw_overlay=True)

    logger.info(f"Detection Status : {analysis.status.value}")
    logger.info(f"Area Ratio       : {analysis.area_ratio * 100:.2f}%")

    if analysis.quadrilateral is not None:
        logger.info(f"Quadrilateral Pts: {analysis.quadrilateral.tolist()}")
        warped = detector.warp_perspective(image, analysis.quadrilateral)
        cfg = get_config()
        warp_path = cfg.PROCESSED_DIR / "page_test_dewarped.jpg"
        cv2.imwrite(str(warp_path), warped)
        logger.info(f"Saved dewarped page to: {warp_path}")

    if analysis.annotated_frame is not None:
        cfg = get_config()
        overlay_path = cfg.PROCESSED_DIR / "page_test_annotated.jpg"
        cv2.imwrite(str(overlay_path), analysis.annotated_frame)
        logger.info(f"Saved annotated preview to: {overlay_path}")


def run_demo_mode() -> None:
    """Runs an automated end-to-end reading demonstration using synthetic test page."""
    logger = get_logger("DemoMode")
    from visionvoice.ocr.engine import OCREngine
    from visionvoice.speech.base import get_tts

    logger.info("==================================================")
    logger.info("   VISIONVOICE AUTOMATED ASSISTIVE READING DEMO   ")
    logger.info("==================================================")

    # 1. Create a synthetic clean book page image
    img = np.full((1080, 1920, 3), 245, dtype=np.uint8)
    # Draw book borders
    cv2.rectangle(img, (200, 100), (1720, 980), (255, 255, 255), -1)
    cv2.rectangle(img, (200, 100), (1720, 980), (180, 180, 180), 2)
    # Book spine line
    cv2.line(img, (960, 100), (960, 980), (220, 220, 220), 2)

    # Put title and text on page
    cv2.putText(img, "Chapter 1: The Magic of Reading", (260, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 3)
    cv2.putText(img, "VisionVoice brings books to life for visually impaired users.", (260, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2)
    cv2.putText(img, "It captures pages automatically when held steady under the camera.", (260, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2)
    cv2.putText(img, "High accuracy optical character recognition extracts the paragraphs.", (260, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2)
    cv2.putText(img, "Natural text-to-speech reads each sentence aloud clearly.", (260, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2)
    cv2.putText(img, "Enjoy reading anytime with hands-free voice control.", (260, 720), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2)

    demo_img_path = get_config().CAPTURES_DIR / "demo_book_page.jpg"
    cv2.imwrite(str(demo_img_path), img)
    logger.info(f"Generated demo page image: {demo_img_path}")

    # 2. Run OCR Pipeline
    logger.info("Executing High-Accuracy OCR Pipeline...")
    engine = OCREngine()
    result = engine.process_image(img)

    logger.info(f"OCR Status: {'SUCCESS' if result.is_valid else 'FAILED'}")
    logger.info(f"Selected Preprocessing Variant: {result.selected_variant}")
    logger.info(f"Quality Score: {result.composite_score:.2f} / 100")
    logger.info(f"Detected Language: {result.detected_script}")
    logger.info(f"Total Sentences Extracted: {len(result.sentences)}")

    # 3. Reading aloud via TTS
    tts = get_tts()
    tts.speak("Demo mode started. Reading captured page.", blocking=True)
    time.sleep(0.3)

    for idx, sentence in enumerate(result.sentences, 1):
        logger.info(f"Reading Sentence [{idx}/{len(result.sentences)}]: {sentence}")
        tts.speak(sentence, language=result.detected_language, blocking=True)
        time.sleep(0.2)

    tts.speak("Demo page finished successfully.", blocking=True)
    logger.info("Demo complete.")


def main() -> None:
    args = parse_arguments()
    cfg = get_config()
    setup_logging(cfg.LOGS_DIR, cfg.DEBUG_MODE)

    if args.camera_test:
        run_camera_test()
    elif args.ocr:
        run_ocr_test(args.ocr, fast_mode=args.fast)
    elif args.demo:
        run_demo_mode()
    elif args.test_tts:
        run_tts_test()
    elif args.test_voice:
        run_voice_test()
    elif args.test_page:
        run_page_detection_test(args.test_page)
    else:
        from visionvoice.core.controller import VisionVoiceController
        controller = VisionVoiceController(show_preview=args.dev or cfg.SHOW_CAMERA_PREVIEW)
        controller.start()


if __name__ == "__main__":
    main()
