# Venice Image Generation Assistant Guide

This document outlines how to use the Venice Image module to generate and edit images by embedding a detail block in your responses.


## Image Operation Workflow
1.  **Act:** Emit the image operation after your conversational response.
2.  **Format:** Emit one HTML `<details>` block for the image operation. The `<summary>` MUST start with "Image:" and specify the operation.

## Generating New Images
<details>
   <summary>Image: New</summary>
   <invoke>
   <action>generate</action>
   <prompt>A futuristic cityscape at sunset, with flying cars and neon signs.</prompt>
   <style_preset>Photographic</style_preset>
   <variants>2</variants>
   <image_size>1280x720</image_size>
   </invoke>
</details>

### Parameters:
-   `action` (str, required): Must be `generate`.
-   `prompt` (str, required): A description of the image to generate. (1-1500 characters).
-   `model` (str, optional): See the [Available Models] section for a list of valid options. Default `hidream`.
-   `style_preset` (str, optional): See the [Available Styles] section for a list of valid options. Default `none`.
-   `negative_prompt` (str, optional): A description of what to avoid in the image. (Max 1500 characters). Default `none`.
-   `variants` (int, optional): The number of images to generate. (Range: 1-4). Defaults to `1`.
-   `image_size` (str, optional): The dimensions of the image (e.g., `1024x1024`). Max width/height is 1280. Defaults to `auto`.
-   `cfg_scale` (float, optional): Prompt adherence. (Range: 0-20). Defaults to `7.5`.
-   `steps` (int, optional): Number of generation steps. (Range: 1-50, model dependent). Defaults to `30`.
-   `seed` (int, optional): A seed for deterministic generation. (Range: -999999999 to 999999999).

### Available Models
| Model ID | Name | Default Steps | Max Steps | Prompt Character Limit |
| `venice-sd35` | Venice SD35 | 25 | 30 | 1500 |
| `hidream` | HiDream | 20 | 50 | 1500 |
| `flux-dev` | FLUX Standard | 25 | 30 | 2048 |
| `flux-dev-uncensored` | FLUX Custom | 25 | 30 | 2048 |
| `lustify-sdxl` | Lustify SDXL | 20 | 50 | 1500 |
| `qwen-image` | Qwen Image | 8 | 8 | 1500 |
| `wai-Illustrious` | Anime (WAI) | 25 | 30 | 1500 |

### Quick Selection Guide
For Text in Images: qwen-image (unmatched text rendering) 
For Photorealism: flux-dev (highest quality, best hands/faces) 
For Complex Prompts: hidream (best prompt adherence and detail) 
For Anime/Manga: wai-Illustrious (specialized anime model) 
For Versatility: venice-sd35 (balanced quality/speed, style flexibility) 
For Adult Content: lustify-sdxl (NSFW-specialized with aesthetic focus) 
For Speed + Quality: qwen-image (8 steps only, still excellent output)

### Available Styles
`3D Model`, `Analog Film`, `Anime`, `Cinematic`, `Comic Book`, `Craft Clay`, `Digital Art`, `Enhance`, `Fantasy Art`, `Isometric Style`, `Line Art`, `Lowpoly`, `Neon Punk`, `Origami`, `Photographic`, `Pixel Art`, `Texture`, `Advertising`, `Food Photography`, `Real Estate`, `Abstract`, `Cubist`, `Graffiti`, `Hyperrealism`, `Impressionist`, `Pointillism`, `Pop Art`, `Psychedelic`, `Renaissance`, `Steampunk`, `Surrealist`, `Typography`, `Watercolor`, `Fighting Game`, `GTA`, `Super Mario`, `Minecraft`, `Pokemon`, `Retro Arcade`, `Retro Game`, `RPG Fantasy Game`, `Strategy Game`, `Street Fighter`, `Legend of Zelda`, `Architectural`, `Disco`, `Dreamscape`, `Dystopian`, `Fairy Tale`, `Gothic`, `Grunge`, `Horror`, `Minimalist`, `Monochrome`, `Nautical`, `Space`, `Stained Glass`, `Techwear Fashion`, `Tribal`, `Zentangle`, `Collage`, `Flat Papercut`, `Kirigami`, `Paper Mache`, `Paper Quilling`, `Papercut Collage`, `Papercut Shadow Box`, `Stacked Papercut`, `Thick Layered Papercut`, `Alien`, `Film Noir`, `HDR`, `Long Exposure`, `Neon Noir`, `Silhouette`, `Tilt-Shift`


## Editing Existing Images
<details>
   <summary>Image: Edit</summary>
   <invoke>
   <action>edit</action>
   <prompt>Make the sky look like a vibrant sunset.</prompt>
   <source_image_url>https://im.di.st/123xyz.webp</source_image_url>
   </invoke>
</details>

### Parameters:
-   `action` (str, required): Must be `edit`.
-   `prompt` (str, required): Instructions for how to edit the image (e.g., "Change the background to a forest"). (Max 1500 characters).
-   `source_image_url` (str, required): The URL of the image to edit.

## Upscaling Images
<details>
   <summary>Image: Upscale</summary>
   <invoke>
   <action>upscale</action>
   <prompt>Upscale this image 4x and enhance the details.</prompt>
   <replication>0.5</replication>
   <source_image_url>https://im.di.st/456abc.webp</source_image_url>
   </invoke>
</details>

### Parameters:
-   `action` (str, required): Must be `upscale`.
-   `source_image_url` (str, required): The URL of the image to upscale.
-   `prompt` (str, optional): A prompt to guide the upscaling process. Can include `2x` or `4x` to specify the factor.
-   `enhance` (bool, optional): Whether to enhance the image during upscaling. Defaults to `false`.
-   `enhance_prompt` (str, optional): A prompt for the enhancement process. (Max 1500 characters).
-   `replication` (float, optional): How strongly lines and noise are preserved. (Range: 0-1). Defaults to 0.35.
-   `scale` (int, optional): The upscale factor. (Range: 1-4). Defaults to 2.

### Notes - Model-Specific Tips:
wai-Illustrious: This model is already optimized for anime generation. Skip the style_preset parameter to avoid over-stylization - the model's inherent training produces cleaner, more natural anime results without additional style layering.