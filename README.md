# ASL Image Upload → Word Builder

This is a web-based tool to **upload ASL (American Sign Language) hand images** and automatically build words from the detected letters. It supports **bulk image uploads**, provides **image previews**, and dynamically displays the detected word with animations.

The backend uses an **AI model fine-tuned on ASL images**, specifically [`microsoft/resnet-34`](https://huggingface.co/microsoft/resnet-34), to predict the letter represented in each hand image.

## Features

- Upload multiple images at once and process them **simultaneously**.
- Preview images as **full pictures** without cropping.
- Dynamically build a **word** from detected letters.
- **Clear Images** and **Clear Word** buttons for easy resets.
- Responsive **multi-column layout** for previews and controls.
- Smooth **GSAP animations** for letters.
- Powered by a **fine-tuned AI model** (`microsoft/resnet-34`) for ASL letter detection.
