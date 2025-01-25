import streamlit as st
import pytube
import os
import whisperx
import torch
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip
from transformers import pipeline
from langdetect import detect

class VideoProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.whisper_model = whisperx.load_model("base", device=self.device)

    def download_video(self, url):
        try:
            yt = pytube.YouTube(url)
            video = yt.streams.filter(progressive=True, file_extension='mp4').first()
            return video.download()
        except Exception as e:
            st.error(f"Error downloading video: {str(e)}")
            return None

    def extract_audio(self, video_path):
        try:
            video = VideoFileClip(video_path)
            audio_path = video_path.replace('.mp4', '.wav')
            video.audio.write_audiofile(audio_path)
            return audio_path
        except Exception as e:
            st.error(f"Error extracting audio: {str(e)}")
            return None

    def transcribe_audio(self, audio_path):
        try:
            result = self.whisper_model.transcribe(audio_path)
            return result["segments"]
        except Exception as e:
            st.error(f"Error transcribing audio: {str(e)}")
            return None

    def get_highlights(self, transcription):
        full_text = " ".join([seg["text"] for seg in transcription])
        language = detect(full_text)
        
        # Generate summary
        summary = self.summarizer(full_text, max_length=200, min_length=50, num_beams=4)
        
        # Extract highlights between 45-59 seconds
        highlights = []
        for segment in transcription:
            if 45 <= segment["start"] <= 59:
                highlights.append({
                    "text": segment["text"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "rating": (59 - segment["start"]) / 14  # Rating based on position in timeframe
                })
        
        # Sort by rating and get top 5
        highlights.sort(key=lambda x: x["rating"], reverse=True)
        return highlights[:5], language

    def create_clip(self, video_path, start_time, end_time, text, output_path):
        video = VideoFileClip(video_path)
        
        # Cut the clip
        clip = video.subclip(start_time, end_time)
        
        # Resize to 9:16 ratio
        w, h = clip.size
        target_w = h * 9 // 16
        x_center = (w - target_w) // 2
        clip = clip.crop(x1=x_center, y1=0, x2=x_center+target_w, y2=h)
        
        # Add caption
        txt_clip = TextClip(text, fontsize=24, color='white', bg_color='black',
                           size=(target_w, None), method='caption')
        txt_clip = txt_clip.set_position(('center', 'bottom')).set_duration(clip.duration)
        
        # Combine video and text
        final_clip = CompositeVideoClip([clip, txt_clip])
        final_clip.write_videofile(output_path, codec='libx264')
        
        return output_path

def main():
    st.title("YouTube Video Processor")
    
    processor = VideoProcessor()
    
    url = st.text_input("Enter YouTube URL:")
    
    if st.button("Process Video"):
        if url:
            with st.spinner("Processing video..."):
                # Download video
                video_path = processor.download_video(url)
                if not video_path:
                    return
                
                # Extract audio
                audio_path = processor.extract_audio(video_path)
                if not audio_path:
                    return
                
                # Get transcription
                transcription = processor.transcribe_audio(audio_path)
                if not transcription:
                    return
                
                # Get highlights and language
                highlights, language = processor.get_highlights(transcription)
                
                # Display highlights
                st.subheader("Top 5 Highlights")
                for i, highlight in enumerate(highlights, 1):
                    st.write(f"Highlight {i}")
                    st.write(f"Text: {highlight['text']}")
                    st.write(f"Time: {highlight['start']:.2f}s - {highlight['end']:.2f}s")
                    st.write(f"Rating: {highlight['rating']:.2f}")
                    
                    # Create clip for each highlight
                    output_path = f"highlight_{i}.mp4"
                    processor.create_clip(
                        video_path,
                        highlight["start"],
                        highlight["end"],
                        highlight["text"],
                        output_path
                    )
                    
                    # Provide download button
                    with open(output_path, "rb") as file:
                        st.download_button(
                            label=f"Download Highlight {i}",
                            data=file,
                            file_name=output_path,
                            mime="video/mp4"
                        )
                
                st.success("Processing complete!")
                
                # Cleanup
                os.remove(video_path)
                os.remove(audio_path)
                for i in range(1, 6):
                    os.remove(f"highlight_{i}.mp4")

if __name__ == "__main__":
    main()