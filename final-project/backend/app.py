from flask import Flask, request, jsonify, send_file
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import StableDiffusionPipeline
import torch
import textwrap
from PIL import Image
from moviepy.editor import ImageSequenceClip, AudioFileClip
from gtts import gTTS
from fpdf import FPDF
import re
import os
from flask_cors import CORS
import base64
from io import BytesIO
# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enables CORS for all routes




class TextImageVideoGenerator:
    def __init__(self, text_model_name='EleutherAI/gpt-neo-1.3B',
                 image_model_name='runwayml/stable-diffusion-v1-5'):
        # Text generation setup
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_model = AutoModelForCausalLM.from_pretrained(text_model_name)

        # Image generation setup
        self.image_pipeline = StableDiffusionPipeline.from_pretrained(
            image_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        if torch.cuda.is_available():
            self.image_pipeline = self.image_pipeline.to("cuda")

    def generate_text(self, prompt, max_length=500):
    
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

    # Generate text with attention mask and explicitly set pad_token_id
        output = self.text_model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        do_sample=True,
        pad_token_id=self.tokenizer.eos_token_id  # Explicitly set the pad_token_id
    )

        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text

    def extract_image_prompts(self, text):
        paragraphs = text.split("\n\n")
        prompts = []
        for paragraph in paragraphs:
            words = re.findall(r'\b\w+\b', paragraph.lower())
            common_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'])
            key_words = [word for word in words if word not in common_words]
            prompts.append(" ".join(key_words[:10]))
        return prompts

    def generate_images(self, prompts):
        images = []
        for prompt in prompts:
            print(f"Generating image for: {prompt}")
            image = self.image_pipeline(
                prompt,
                num_inference_steps=25,  # Reduced steps
                height=256,  # Lower resolution
                width=256
        ).images[0]
        images.append(image)
        return images

    def save_text_and_images_as_pdf(self, text, images, filename="generated_content.pdf"):
        try:
            print(f"Saving PDF to {filename}...")
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", size=12)

        # Add text content
            for line in text.split("\n"):
                pdf.multi_cell(0, 10, line)

        # Add images
            for idx, img in enumerate(images):
                pdf.add_page()  # Add a new page for each image
                print(f"Adding image {idx + 1} to PDF...")
            
            # Save image temporarily
                temp_image_path = f"temp_image_{idx}.png"
                img.save(temp_image_path)
            
            # Calculate dimensions to fit the page
                pdf_width = 190  # Width for A4 page in mm minus margins
                pdf_image = Image.open(temp_image_path)
                width, height = pdf_image.size
                aspect_ratio = height / width
                pdf_height = pdf_width * aspect_ratio

            # Add image to the PDF
                pdf.image(temp_image_path, x=10, y=pdf.get_y() + 10, w=pdf_width, h=pdf_height)

            # Clean up temporary image
                os.remove(temp_image_path)

        # Save PDF
            pdf.output(filename)
            print(f"PDF saved successfully: {filename}")
            return filename
        except Exception as e:
            print(f"Error while generating PDF with text and images: {e}")
            raise

    def generate_audio(self, text, filename="narration.mp3"):
        print("Generating audio narration...")
        tts = gTTS(text=text, lang='en')
        tts.save(filename)
        return filename

    def generate_video(self, images, audio_file, output_file="output_video.mp4", fps=1):
        print("Creating video from images and audio...")
        image_files = []
        for idx, img in enumerate(images):
            img_file = f"temp_image_{idx}.png"
            img.save(img_file)
            image_files.append(img_file)
        clip = ImageSequenceClip(image_files, fps=fps)
        audio = AudioFileClip(audio_file)
        video = clip.set_audio(audio)
        video.write_videofile(output_file, codec="libx264", audio_codec="aac")
        for img_file in image_files:
            os.remove(img_file)
        return output_file

# Initialize the generator
generator = TextImageVideoGenerator()



@app.route('/generate', methods=['POST'])
def generate_content():
    data = request.json
    prompt = data.get('prompt', '')

    if not prompt:
        return jsonify({'error': 'Prompt is required.'}), 400

    try:
        # Generate text
        print("Generating text...")
        generated_text = generator.generate_text(prompt)
        print("Generated text successfully.")
        
        # Extract image prompts
        print("Extracting image prompts...")
        image_prompts = generator.extract_image_prompts(generated_text)
        print("Image prompts:", image_prompts)
        
        # Generate images
        print("Generating images...")
        images = generator.generate_images(image_prompts)
        print("Generated images successfully.")
        
        # Convert images to Base64 strings
        base64_images = []
        for image in images:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
            base64_images.append(base64_image)

        return jsonify({
            'text': generated_text,
            'images': base64_images,
        })

    except Exception as e:
        print("Error occurred:", e)
        return jsonify({'error': str(e)}), 500    

@app.route('/download/<path:filename>', methods=['GET'])
def download_file(filename):
    """
    Route to download generated files.
    """
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
