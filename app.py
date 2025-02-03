import os
import re
import torch
import zipfile
import requests
import fal_client
import streamlit as st
from PIL import Image
from io import BytesIO
import google.generativeai as genai
from groq import Groq
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from concurrent.futures import ThreadPoolExecutor
from transformers import BlipProcessor, BlipForConditionalGeneration
# from dotenv import load_dotenv

# Load environment variables
# load_dotenv()

# Cache BLIP model with optimized settings
@st.cache_resource
def load_blip_components():
    # Load processor and model together
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # Determine device first
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Load model with appropriate dtype
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        torch_dtype=torch_dtype
    ).to(device)

    return processor, model

# Initialize processor and model together
blip_processor, blip_model = load_blip_components()

def validate_api_keys(required_services):
    errors = []
    key_checks = {
        "Gemini": not st.session_state.gemini_key.startswith("Enter your"),
        "DeepSeek": not st.session_state.deepseek_key.startswith("Enter your"),
        "Flux": not st.session_state.flux_key.startswith("Enter your")
    }
    
    for service in required_services:
        if not key_checks.get(service, False):
            errors.append(f"Please enter your {service} API key in the 🔑 API Key Management section")
    
    if errors:
        for error in errors:
            st.error(error)
        return False
    return True

# Optimized caption generation
def generate_caption(image):
    # Use the cached processor and model
    inputs = blip_processor(images=image.convert("RGB"), return_tensors="pt").to(blip_model.device)
    with torch.inference_mode():
        caption_ids = blip_model.generate(
            **inputs,
            max_length=50,
            num_beams=4,
            early_stopping=True
        )
    return blip_processor.decode(caption_ids[0], skip_special_tokens=True)

# Optimized image processing
def process_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img.thumbnail((512, 512))  # Faster than resize with maintained aspect ratio
    return img

# Parallel image generation for Story to Image mode
def generate_single_image(args):
    part, style = args
    image_prompt = f"Generate image for: '{part}' with {style}-theme"
    try:
        result = fal_client.subscribe(
            "fal-ai/flux/schnell",
            arguments={"prompt": image_prompt},
            with_logs=False
        )
        if "images" in result:
            return requests.get(result["images"][0]["url"]).content
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
    return None

# Cache AI model initialization
@st.cache_resource
def init_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("models/gemini-1.5-flash")

@st.cache_resource
def init_deepseek(api_key):
    return Groq(api_key=api_key)

# AI Story Generation Functions
def generate_story_gemini(model, caption, n, theme):
    prompt = f"""Write a complete story of exactly {n} words about {caption} with a {theme} theme.
    Story structure: Beginning, Middle, Conclusion"""
    result = model.generate_content(prompt)
    return result.candidates[0].content.parts[0].text

def generate_story_deepseek(client, caption, n, theme):
    prompt = f"Write a complete story of exactly {n}-words about {caption} with a {theme} theme."
    completion = client.chat.completions.create (
        model="deepseek-r1-distill-llama-70b",
        messages=[
            {"role": "user", 
             "content": prompt}
        ],
        temperature=1.2,
        stop=["```"],
        top_p=0.95,
        reasoning_format="raw"
    )
    
    full_story = completion.choices[0].message.content
    # Remove any unwanted <think> tags
    clean_story = re.sub(r"<think>.*?</think>", "", full_story, flags=re.DOTALL)

    return clean_story.strip()

# Streamlit UI
st.title("֍ StoryBoard AI")
st.markdown("<h4 style='font-size: 18px; font-weight: normal;'>Elevate your storytelling to new heights with the limitless power of AI!</h4>", unsafe_allow_html=True)

# Add API key management button in top right
col1, col2 = st.columns([4, 1])
with col2:
    if st.button("🔑 Use Custom API Keys"):
        st.session_state.show_api_keys = not st.session_state.get('show_api_keys', False)

# API Key Input Section
if st.session_state.get('show_api_keys', False):
    with st.expander("🔑 API Key Management", expanded=True):
        # DeepSeek API Key
        ds_key = st.text_input("DeepSeek API Key", 
                             value=st.session_state.get('deepseek_key', ''),
                             type="password")
        st.markdown("<a href='https://console.groq.com/keys' target='_blank'>Create DeepSeek Key</a>", 
                   unsafe_allow_html=True)
        
        # Gemini API Key
        gem_key = st.text_input("Gemini API Key", 
                              value=st.session_state.get('gemini_key', ''),
                              type="password")
        st.markdown("<a href='https://aistudio.google.com/app/apikey' target='_blank'>Create Gemini Key</a>", 
                   unsafe_allow_html=True)
        
        # Flux API Key
        flux_key = st.text_input("Flux API Key", 
                               value=st.session_state.get('flux_key', ''),
                               type="password")
        st.markdown("<a href='https://fal.ai/dashboard/keys' target='_blank'>Get Flux Key</a>", 
                   unsafe_allow_html=True)
        
        # Save keys to session state
        if st.button("Save Keys"):
            st.session_state.deepseek_key = ds_key
            st.session_state.gemini_key = gem_key
            st.session_state.flux_key = flux_key
            os.environ["FAL_KEY"] = flux_key  # Update Flux key in environment
            st.success("API keys saved for this session!")

# API Keys (use custom keys if available, otherwise .env)
gemini_api_key = st.session_state.get('gemini_key', "Enter your Gemini API key here")
deepseek_api_key = st.session_state.get('deepseek_key', "Enter your DeepSeek API key here")
os.environ["FAL_KEY"] = st.session_state.get('flux_key', "Enter your Flux API key here")

# Dropdown Menu for Functionality Selection
options = ["Image to Story", "Storyboard", "Story to Image"]
selected_option = st.selectbox("Select Mode", options, index=0)

# Image to Story
if selected_option == "Image to Story":
    st.header("🖼️ Image to Story")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        resized_image = image.resize((384, 384)) # Resize for efficiency
        st.image(image, caption="Uploaded Image", use_container_width=True)
        caption = generate_caption(resized_image)
        st.text_area("Edit Caption", caption, key="caption")

    story_length = st.slider("Select story length (words)", 200, 800, 300, 15)
    theme = st.selectbox("Select a theme", ["Fantasy", "Sci-Fi", "Mystery", "Romance", "Adventure", "Custom"])
    if theme == "Custom":
        theme = st.text_input("Enter your custom theme", "")

    model_choice = st.selectbox("Select AI model", ["DeepSeek", "Gemini"], index=0)

    if st.button("Generate Story") and "caption" in st.session_state:
        required_services = ["Gemini"] if model_choice == "Gemini" else ["DeepSeek"]
        if not validate_api_keys(required_services):
            st.stop()
        caption = st.session_state["caption"]
        with st.spinner("Generating story... Please wait."):
            if model_choice == "Gemini":
                model = init_gemini(gemini_api_key)
                story = generate_story_gemini(model, caption, story_length, theme)
            else:
                client = init_deepseek(deepseek_api_key)
                story = generate_story_deepseek(client, caption, story_length, theme)

        st.subheader(f"📖 Generated Story ({model_choice})")
        st.write(story)

        # Download Button
        st.download_button("Download Story", data=story, file_name="generated_story.txt", mime="text/plain")

# Storyboard
elif selected_option == "Storyboard":
    st.header("🎞️ Storyboard - Upload Multiple Images")
    uploaded_images = st.file_uploader("Upload multiple images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    if uploaded_images:
        captions = []
        for img_file in uploaded_images:
            img = Image.open(img_file)
            resized_img = img.resize((384, 384))  # Resize for efficiency
            st.image(img, caption=img_file.name, use_container_width=True)
            captions.append(generate_caption(resized_img))
        st.text_area("Edit Captions", "\n".join(captions), key="storyboard_captions")

    story_length = st.slider("Select story length (words)", 500, 1500, 600, 25)
    theme = st.selectbox("Select a theme", ["Fantasy", "Sci-Fi", "Mystery", "Romance", "Adventure", "Custom"], key="storyboard_theme")
    if theme == "Custom":
        theme = st.text_input("Enter your custom theme", "", key="custom_theme")

    model_choice = st.selectbox("Select AI model", ["DeepSeek", "Gemini"], index=0, key="storyboard_model")

    if st.button("Generate Story") and "storyboard_captions" in st.session_state:
        required_services = ["Gemini"] if model_choice == "Gemini" else ["DeepSeek"]
        if not validate_api_keys(required_services):
            st.stop()
        captions = st.session_state["storyboard_captions"].split("\n")
        combined_caption = " ".join(captions)

        with st.spinner("Generating story... Please wait."):
            if model_choice == "Gemini":
                model = init_gemini(gemini_api_key)
                story = generate_story_gemini(model, combined_caption, story_length, theme)
            else:
                client = init_deepseek(deepseek_api_key)
                story = generate_story_deepseek(client, combined_caption, story_length, theme)

        st.subheader(f"📖 Generated Story ({model_choice})")
        st.write(story)
        st.download_button("Download Story", data=story, file_name="storyboard_story.txt", mime="text/plain")


# Story to Image - Optimized Section
elif selected_option == "Story to Image":
    st.header("📝 Story to Image")
    story = st.text_area("Enter your story description")
    
    art_styles = ["Realistic", "Anime", "Watercolor", "Cyberpunk", "Pixel Art", "Custom"]
    selected_style = st.selectbox("Select an art style", art_styles)
    if selected_style == "Custom":
        selected_style = st.text_input("Enter custom art style")
    
    num_images = st.slider("Select number of images", 1, 10, 3)
    
    if st.button("Generate Images") and story:
        if not validate_api_keys(["Gemini", "Flux"]):
            st.stop()
        st.session_state.generated_images = []
        st.session_state.image_prompts = []
        
        # Generate story parts with optimized prompt
        gemini_model = init_gemini(gemini_api_key)
        prompt = f"Split this story into {num_images} concise parts separated by '|||': {story}"
        result = gemini_model.generate_content(prompt)
        story_parts = [p.strip() for p in result.text.split("|||") if p.strip()]
        
        # Parallel image generation
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for part in story_parts[:num_images]:
                image_prompt = f"Generate image for: '{part}' with {selected_style}-theme"
                futures.append(executor.submit(generate_single_image, (part, selected_style)))
                st.session_state.image_prompts.append(part)
            
            progress_bar = st.progress(0)
            for i, future in enumerate(futures):
                image_data = future.result()
                if image_data:
                    img = Image.open(BytesIO(image_data))
                    
                    # Display image with prompt
                    with st.container():
                        st.image(img, caption=f"Image {i+1}", use_container_width=True)
                        st.caption(f"{st.session_state.image_prompts[i]}")
                    
                    st.session_state.generated_images.append((f"image_{i+1}.jpg", image_data))
                progress_bar.progress((i+1)/num_images)

    # Download options
    if "generated_images" in st.session_state and st.session_state.generated_images:
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        # ZIP Download
        with col1:
            with BytesIO() as zip_buffer:
                with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                    for filename, data in st.session_state.generated_images:
                        zip_file.writestr(filename, data)
                st.download_button(
                    label="📥 Download Images (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="generated_images.zip",
                    mime="application/zip",
                    help="Download all generated images as ZIP archive"
                )
        
        # PDF Download
        with col2:

            pdf_buffer = BytesIO()
            c = canvas.Canvas(pdf_buffer, pagesize=letter)
            width, height = letter
            
            for idx, (filename, data) in enumerate(st.session_state.generated_images):
                img = ImageReader(BytesIO(data))
                img_width, img_height = img.getSize()
                aspect = img_height / float(img_width)
                c.setFont("Helvetica", 12)
                
                # Add prompt text
                c.drawString(50, height - 50, f"Image {idx+1} ")
                text = c.beginText(50, height - 70)
                text.setFont("Helvetica", 10)
                text.textLines(st.session_state.image_prompts[idx])
                c.drawText(text)
                
                # Add image
                img_width = width - 100
                img_height = img_width * aspect
                c.drawImage(img, 50, height - 70 - img_height - 50, 
                          width=img_width, height=img_height)
                c.showPage()
            
            c.save()
            st.download_button(
                label="📄 Download Story PDF",
                data=pdf_buffer.getvalue(),
                file_name="story_images.pdf",
                mime="application/pdf",
                help="Download PDF with images and their prompts"
            )