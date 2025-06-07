import tkinter as tk
import customtkinter as ctk
import soundfile as sf
import sounddevice as sd
import whisper
import os

# Load Whisper model
model = whisper.load_model("small")

# Set appearance and theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Initialize main window
app = ctk.CTk()
app.geometry("600x750")
app.title("🎤 AI Voice to Text Converter")

# Header Label
header = ctk.CTkLabel(
    master=app,
    text="🎙️ AI Voice to Text & Translation",
    font=("Segoe UI Bold", 24),
    text_color="#00ffff"
)
header.pack(pady=20)

# Recording status label
main_label = ctk.CTkLabel(
    master=app,
    text="Click 'Record' to start recording...",
    font=("Segoe UI", 16),
    text_color="white"
)
main_label.pack(pady=10)

# Output box
output_box = ctk.CTkTextbox(
    master=app,
    width=540,
    height=400,
    font=("Consolas", 14),
    text_color="white",
    fg_color="#1e1e1e",
    wrap="word",
    corner_radius=12,
    border_width=2,
    border_color="#00ffff"
)
output_box.pack(pady=10)

# Record audio
def voice_rec():
    fs = 48000
    duration = 5
    main_label.configure(text="🔴 Recording in progress...")
    app.update_idletasks()

    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    sf.write("my_Audio_file.flac", myrecording, fs)
    main_label.configure(text="✅ Recording complete.")

# Transcribe audio
def transcribe():
    audio = "my_Audio_file.flac"
    if not os.path.exists(audio):
        main_label.configure(text="⚠️ Please record audio first.")
        return

    main_label.configure(text="🕐 Transcribing...")
    results = model.transcribe(audio, fp16=False, task="transcribe")
    output_box.delete("1.0", tk.END)
    output_box.insert(tk.END, results["text"])
    main_label.configure(text="✅ Transcription complete.")

# Translate audio
def translate():
    audio = "my_Audio_file.flac"
    if not os.path.exists(audio):
        main_label.configure(text="⚠️ Please record audio first.")
        return

    main_label.configure(text="🕐 Translating...")
    results = model.transcribe(audio, fp16=False, task="translate")
    output_box.delete("1.0", tk.END)
    output_box.insert(tk.END, results["text"])
    main_label.configure(text="🌐 Translation complete.")

# Buttons Frame
button_frame = ctk.CTkFrame(master=app, fg_color="transparent")
button_frame.pack(pady=20)

# Record Button
recordButton = ctk.CTkButton(
    master=button_frame,
    text="🎤 Record",
    command=voice_rec,
    width=160,
    height=50,
    font=("Segoe UI", 16),
    corner_radius=10
)
recordButton.grid(row=0, column=0, padx=15)

# Transcribe Button
transcribeButton = ctk.CTkButton(
    master=button_frame,
    text="📝 Transcribe",
    command=transcribe,
    width=160,
    height=50,
    font=("Segoe UI", 16),
    corner_radius=10
)
transcribeButton.grid(row=0, column=1, padx=15)

# Translate Button
translateButton = ctk.CTkButton(
    master=button_frame,
    text="🌍 Translate",
    command=translate,
    width=160,
    height=50,
    font=("Segoe UI", 16),
    corner_radius=10
)
translateButton.grid(row=0, column=2, padx=15)

# Run the app
app.mainloop()
