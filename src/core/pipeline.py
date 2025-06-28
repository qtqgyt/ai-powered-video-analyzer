# src/core/pipeline.py
import logging
from typing import Optional

from ai_models.yolo_detector import YOLODetector
from .video_utils import extract_audio, extract_frames_for_analysis # etc.
from src.ai_models.whisper_transcriber import WhisperTranscriber
from src.ai_models.ollama_summarizer import OllamaSummarizer
#from src.ai_models.yolo_detector import YOLODetector
#from src.ai_models.blip_captioner import BLIPCaptioner
#from src.ai_models.pann_audio_event_detector import PANNAudioEventDetector
#from src.ai_models.llm_summarizer import LLMSummarizer
from src.config import settings # For model paths, etc.
from pathlib import Path

# (Or, model instances can be passed in during __init__)

logger = logging.getLogger(__name__)

class VideoAnalysisPipeline:
    def __init__(self, video_path, transcription_language="en", summarization_model="gemma3",
                 perform_object_detection=True, perform_scene_description=True,
                 perform_transcription=True, perform_audio_events=True, perform_summarization=True):
        # Convert video_path to Path object and resolve it
        self.video_path = str(Path(video_path).resolve())

        self.transcription_language = transcription_language
        self.summarization_model_name = summarization_model

        self.perform_object_detection = perform_object_detection
        self.perform_scene_description = perform_scene_description
        self.perform_transcription = perform_transcription
        self.perform_audio_events = perform_audio_events
        self.perform_summarization = perform_summarization

        # Initialize models (consider lazy loading or explicit setup method)
        if self.perform_transcription:
            self.transcriber = WhisperTranscriber(model_path=settings.WHISPER_MODEL_PATH)
        if self.perform_object_detection:
            self.object_detector = YOLODetector(model_path=settings.YOLO_MODEL_PATH)
        # if self.perform_scene_description:
        #     self.captioner = BLIPCaptioner(model_path=settings.BLIP_MODEL_PATH)
        # if self.perform_audio_events:
        #     self.audio_event_detector = PANNAudioEventDetector(model_path=settings.PANNS_MODEL_PATH)
        self.summarizer = OllamaSummarizer(model_name=summarization_model)

    def run_analysis(self):
        logger.info(f"Pipeline started for {self.video_path}")
        results = {}

        # 1. Extract Audio            
        audio_path = extract_audio(self.video_path) 
        if not audio_path:
            logger.error("Failed to extract audio.")
            return results

        # 2. Speech Transcription (Whisper)
        if self.perform_transcription and hasattr(self, 'transcriber'):
            logger.info("Transcribing speech...")
            results["transcription"] = self.transcriber.transcribe(audio_path, language=self.transcription_language)
            logger.debug(f"Transcription (partial): {results['transcription'][:100]}...")

    #     # 3. Audio Event Detection (PANNs)
    #     if self.perform_audio_events and hasattr(self, 'audio_event_detector'):
    #         raise NotImplementedError("This is not yet implemented.")
    #         logger.info("Detecting audio events...")
    #         results["audio_events"] = self.audio_event_detector.detect_events(audio_path)
    #         logger.debug(f"Audio Events: {results['audio_events']}")

        # 4. Frame Extraction (for YOLO & BLIP)
    #     # This needs careful implementation: select keyframes or process at intervals
    #     raise NotImplementedError("This is not yet implemented.")
        logger.info("Extracting frames for visual analysis...")
        frames_for_analysis = extract_frames_for_analysis(self.video_path, interval_seconds=5)

        # 5. Object Detection (YOLO)
        if self.perform_object_detection and hasattr(self, 'object_detector') and frames_for_analysis:
            logger.info("Detecting objects in frames...")
            all_objects = []
            for frame in frames_for_analysis:
                objects = self.object_detector.detect(frame.image)
                if objects:
                    all_objects.append({
                        "timestamp": frame.timestamp,
                        "objects": objects
                    })
            
            # Calculate statistics across all frames
            if all_objects:
                results["object_detections"] = all_objects
                total_objects = sum(len(frame["objects"]) for frame in all_objects)
                unique_objects = len({obj["class_name"] for frame in all_objects for obj in frame["objects"]})
                logger.info(f"Object Detection: Found {total_objects} instances of {unique_objects} unique object types")
            else:
                logger.debug("Object Detection: No objects detected")

    #     # 6. Scene Description (BLIP)
    #     if self.perform_scene_description and hasattr(self, 'captioner') and frames_for_analysis:
    #         raise NotImplementedError("This is not yet implemented.")
    #         logger.info("Generating scene descriptions...")
    #         all_captions = []
    #         for timestamp, frame_image in frames_for_analysis:
    #             caption = self.captioner.caption(frame_image)
    #             if caption:
    #                 all_captions.append({"timestamp": timestamp, "caption": caption})
    #         results["scene_descriptions"] = all_captions
    #         logger.debug(f"Scene Descriptions (sample): {all_captions[:2]}")

        # 7. AI Summarization (LLM via Ollama)
        if self.perform_summarization and hasattr(self, 'summarizer'):
            try:
                logger.info("Generating AI summary...")
                summary_context = self._prepare_summary_context(results)
                
                if not summary_context:
                    logger.warning("No content available for summarization")
                    results["summary"] = "Insufficient content for video summary."
                else:
                    results["summary"] = self.summarizer.summarize(summary_context)
                    if results["summary"]:
                        logger.debug(f"Generated summary ({len(results['summary'])} chars)")
                    else:
                        logger.error("Failed to generate summary")
                        results["summary"] = "Summary generation failed."
            except Exception as e:
                logger.error(f"Error during summarization: {e}")
                results["summary"] = f"Error generating summary: {str(e)}"

        # 8. (Optional) Create Annotated Video & Final Report String
        # This would involve more complex logic in output_generator.py
        results["text_report"] = self._generate_text_report(results)
        # results["annotated_video_path"] = self.output_generator.create_annotated_video(...)
        logger.info("Pipeline finished.")
        return results

    def _prepare_summary_context(self, analysis_results):
        """Prepare a context string for the summarizer from all available analysis results."""
        context_parts = []
        
        # Add transcription if available
        if analysis_results.get("transcription"):
            context_parts.append(f"Speech Content:\n{analysis_results['transcription']}\n")
        
        # Add object detection summary if available
        if analysis_results.get("object_detections"):
            # Group objects by type for a cleaner summary
            object_counts = {}
            for frame in analysis_results["object_detections"]:
                for obj in frame["objects"]:
                    name = obj["class_name"]
                    object_counts[name] = object_counts.get(name, 0) + 1
            
            # Format object detection summary
            if object_counts:
                context_parts.append("Visual Elements Detected:")
                for obj_name, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True):
                    context_parts.append(f"- {obj_name}: {count} occurrences")
        
        # Combine all parts with proper spacing
        return "\n\n".join(context_parts)

    def _generate_text_report(self, analysis_results):
        # Format all collected results into a comprehensive string
        report_parts = [f"Analysis Report for: {self.video_path}\n{'='*40}\n"]
        
        # AI Summary section
        if analysis_results.get("summary"):
            report_parts.append(f"AI Summary:\n{analysis_results['summary']}\n{'-'*40}\n")
        
        # Transcription section
        if analysis_results.get("transcription"):
            report_parts.append(f"Full Transcription:\n{analysis_results['transcription']}\n{'-'*40}\n")
        
        # Object Detection section
        if analysis_results.get("object_detections"):
            report_parts.append("Object Detection Summary:\n")
            
            # Collect all objects and their confidences
            all_detected_objects = {}
            for detection in analysis_results["object_detections"]:
                for obj in detection["objects"]:
                    name = obj["class_name"]
                    conf = obj["confidence"]
                    if name not in all_detected_objects:
                        all_detected_objects[name] = []
                    all_detected_objects[name].append(conf)
            
            # Sort by frequency and get top 5
            top_objects = sorted(
                all_detected_objects.items(), 
                key=lambda x: len(x[1]), 
                reverse=True
            )[:5]
            
            # Calculate statistics for each object type
            report_parts.append("\nTop 5 detected objects:\n")
            for obj_name, confidences in top_objects:
                count = len(confidences)
                mean_conf = sum(confidences) / count
                median_conf = sorted(confidences)[len(confidences)//2]
                skew = abs(mean_conf - median_conf)
                
                report_parts.append(
                    f"  - {obj_name}: {count} occurrences\n"
                    f"    Confidence: mean={mean_conf:.1%}, median={median_conf:.1%}"
                    f"{' (high variance)' if skew > 0.1 else ''}\n"
                )
            
            report_parts.append(f"\n{'-'*40}\n")
        
        return "".join(report_parts)

    def generate_summary(self, report_path: str) -> Optional[str]:
        """Generate a summary of the video analysis report."""
        if not self.perform_summarization:
            logger.debug("Summarization not enabled")
            return None
            
        logger.info("Generating video summary...")
        summary = self.summarizer.summarize_report(report_path)
        
        if summary:
            # Save summary to file
            summary_path = Path(report_path).parent / "video_description.txt"
            try:
                with open(summary_path, "w", encoding="utf-8") as f:
                    f.write("Video Narrative Summary:\n")
                    f.write(summary)
                logger.info(f"Summary saved to: {summary_path}")
                return str(summary_path)
            except Exception as e:
                logger.error(f"Failed to save summary: {e}")
                
        return None