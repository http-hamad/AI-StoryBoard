# StoryBoard AI

## Overview
StoryBoard AI is an **efficient and powerful Streamlit-based application** that leverages the latest AI models to convert images into stories, generate storyboards, and create AI-generated images from text. This project utilizes state-of-the-art models like **DeepSeek**, **Gemini**, and **Flux AI**, ensuring high-quality outputs while maintaining **optimized performance** through caching, parallel processing, and lightweight image handling.

---

## Features

- **Image to Story** – Upload an image, get an AI-generated caption, and turn it into a compelling short story.
- **Storyboard Mode** – Upload multiple images and generate a coherent story based on them.
- **Story to Image** – Provide a text-based story, and the system will generate AI-powered visuals based on its content.
- **Efficient Model Execution** – Uses **optimized caching and GPU/CPU detection** for faster AI processing.
- **Parallel Image Generation** – Uses **ThreadPoolExecutor** to speed up AI-generated images in Story-to-Image mode.
- **Custom API Key Management** – Allows users to input their own API keys for better flexibility and security.

---

## Cutting-Edge AI Models Used

### Text Generation
- **DeepSeek [(deepseek-r1-distill-llama-70b)](https://console.groq.com/playground?model=deepseek-r1-distill-llama-70b)** – Provides high-quality, long-form AI story generation with natural storytelling capabilities.  
- **Google Gemini [(gemini-1.5-flash)](https://gemini.google.com/app)** – A fast and efficient model that generates structured, creative stories.

### Image Captioning
- **BLIP [(Salesforce/blip-image-captioning-base)](https://huggingface.co/Salesforce/blip-image-captioning-base)** – A highly optimized model that generates accurate captions for uploaded images.

### AI-Generated Art
- **Flux AI [(fal-ai/flux/schnell)](https://fal.ai/models/fal-ai/flux/schnell)** – A next-generation model for fast, high-quality AI image generation.

---

## Performance Optimizations

### Efficient AI Model Loading
- Uses `@st.cache_resource` to **cache models** and avoid redundant reloading.
- Automatically **selects the best device** (CUDA, MPS, or CPU) for optimal performance.

### Optimized Image Processing
- Uses `thumbnail(512, 512)` for **faster image handling** while maintaining aspect ratio.
- Resizes input images (`resize(384, 384)`) to speed up processing.

### Parallel Image Generation
- **Uses `ThreadPoolExecutor`** for concurrent image generation, reducing latency in Story-to-Image mode.

### API Response Handling & Caching
- AI-generated images are **cached** to reduce redundant API calls.
- Error handling mechanisms prevent crashes due to failed API requests.

---

## API Keys & Setup
To use this project, you need to generate API keys for the following services:

### Google Gemini API Key
[Get a Gemini API Key](https://aistudio.google.com/app/apikey)

### DeepSeek API Key
[Get a DeepSeek API Key](https://console.groq.com/keys)

### Flux AI API Key
[Get a Flux AI API Key](https://fal.ai/dashboard/keys)

Once you have your API keys, you can enter them in the **Streamlit UI** or save them in an `.env` file.

```bash
GEMINI_API_KEY=your_gemini_key
DEEPSEEK_API_KEY=your_deepseek_key
FAL_KEY=your_flux_key
```

---

## Installation & Running the Project

### Prerequisites
- Python 3.8+
- Pip
- Virtual environment (optional but recommended)

### Install Dependencies
```bash
git clone https://github.com/yourusername/AI-Story-Generator.git
cd AI-Story-Generator
pip install -r requirements.txt
```

### Run the Application
```bash
streamlit run app.py
```

---

## Usage Guide

### Image to Story
1. Upload an image.
2. AI generates a caption using **BLIP**.
3. Choose a theme (Fantasy, Sci-Fi, Mystery, etc.).
4. Select an AI model (**DeepSeek or Gemini**).
5. Generate and download your AI-powered story.

### Storyboard
1. Upload multiple images.
2. AI captions each image.
3. Enter a custom theme and story length.
4. Generate a story connecting all images.

### Story to Image
1. Enter a story description.
2. Choose an art style (Realistic, Anime, Cyberpunk, etc.).
3. Select the number of images.
4. AI generates visuals in parallel.
5. Download the images as a ZIP file.

---

## Project Architecture
```
📂 AI-Story-Generator/
├── 📄 app.py             # Streamlit UI & Core Logic
├── 📄 requirements.txt   # Dependencies
├── 📂 models/            # AI Model Handlers
├── 📂 utils/             # Helper Functions (Processing, API Calls, etc.)
└── 📄 README.md         # Documentation
```

![StoryBoard AI](https://github.com/user-attachments/assets/edecb82d-39f4-44bc-8e5f-64b8a34a02da)
---

## Future Improvements
- Support **for DALL·E 3 for even better image generation.**
- Enhancing **AI story coherence using fine-tuned GPT models.**
- Implement **PDF & video export features.**
- Enhance **AI model selection for user customization.**
- Improve **multi-threaded efficiency for real-time generation.**

---

## License
This project is open-source under the **MIT License**.

---

## Contributing
Feel free to open issues, submit PRs, and improve the project.

You can email me **@contact.mubashirhamad@gmail.com**




**Elevate your storytelling to new heights with the limitless power of AI!**
