curl --request POST \
  --url https://api.venice.ai/api/v1/image/generate \
  --header 'Authorization: Bearer <token>' \
  --header 'Content-Type: application/json' \
  --data '{
  "cfg_scale": 7.5,
  "embed_exif_metadata": false,
  "format": "webp",
  "height": 1024,
  "hide_watermark": false,
  "inpaint": "<any>",
  "lora_strength": 50,
  "model": "hidream",
  "negative_prompt": "Clouds, Rain, Snow",
  "prompt": "A beautiful sunset over a mountain range",
  "return_binary": false,
  "variants": 3,
  "safe_mode": false,
  "seed": 123456789,
  "steps": 20,
  "style_preset": "3D Model",
  "width": 1024
}'

{
  "id": "generate-image-1234567890",
  "images": [
    "<string>"
  ],
  "request": "<any>",
  "timing": {
    "inferenceDuration": 123,
    "inferencePreprocessingTime": 123,
    "inferenceQueueTime": 123,
    "total": 123
  }
}

Generate Images
Generate an image based on input parameters
​
Authorization
stringheaderrequired
Bearer authentication header of the form Bearer <token>, where <token> is your auth token.

Headers
​
Accept-Encoding
string
Supported compression encodings (gzip, br). Only applied when return_binary is false.

Example:
"gzip, br"

Body
application/json
​
model
stringrequired
The model to use for image generation.

Example:
"hidream"

​
prompt
stringrequired
The description for the image. Character limit is model specific and is listed in the promptCharacterLimit setting in the model list endpoint.

Required string length: 1 - 1500
Example:
"A beautiful sunset over a mountain range"

​
cfg_scale
number
CFG scale parameter. Higher values lead to more adherence to the prompt.

Required range: 0 < x <= 20
Example:
7.5

​
embed_exif_metadata
booleandefault:false
Embed prompt generation information into the image's EXIF metadata.

Example:
false

​
format
enum<string>default:webp
The image format to return. WebP are smaller and optimized for web use. PNG are higher quality but larger in file size.

Available options: jpeg, png, webp 
Example:
"webp"

​
height
integerdefault:1024
Height of the generated image. Each model has a specific height and width divisor listed in the widthHeightDivisor constraint in the model list endpoint.

Required range: 0 < x <= 1280
Example:
1024

​
hide_watermark
booleandefault:false
Whether to hide the Venice watermark. Venice may ignore this parameter for certain generated content.

Example:
false

​
inpaint
anydeprecated
This feature is deprecated and was disabled on May 19th, 2025. A revised in-painting API will be launched in the near future.

​
lora_strength
integer
Lora strength for the model. Only applies if the model uses additional Loras.

Required range: 0 <= x <= 100
Example:
50

​
negative_prompt
string
A description of what should not be in the image. Character limit is model specific and is listed in the promptCharacterLimit constraint in the model list endpoint.

Maximum length: 1500
Example:
"Clouds, Rain, Snow"

​
return_binary
booleandefault:false
Whether to return binary image data instead of base64.

Example:
false

​
variants
integer
Number of images to generate (1–4). Only supported when return_binary is false.

Required range: 1 <= x <= 4
Example:
3

​
safe_mode
booleandefault:true
Whether to use safe mode. If enabled, this will blur images that are classified as having adult content.

Example:
false

​
seed
integerdefault:0
Random seed for generation. If not provided, a random seed will be used.

Required range: -999999999 <= x <= 999999999
Example:
123456789

​
steps
integerdefault:20
Number of inference steps. The following models have reduced max steps from the global max: venice-sd35: 30 max steps, hidream: 50 max steps, flux.1-krea: 30 max steps, flux-dev: 30 max steps, flux-dev-uncensored: 30 max steps, lustify-sdxl: 50 max steps, lustify-v7: 25 max steps, pony-realism: 50 max steps, qwen-image: 8 max steps, stable-diffusion-3.5: 30 max steps, wai-Illustrious: 30 max steps. These constraints are exposed in the model list endpoint for each model.

Required range: 0 < x <= 50
Example:
20

​
style_preset
string
An image style to apply to the image. Visit https://docs.venice.ai/api-reference/endpoint/image/styles for more details.

Example:
"3D Model"

​
width
integerdefault:1024
Width of the generated image. Each model has a specific height and width divisor listed in the widthHeightDivisor constraint in the model list endpoint.

Required range: 0 < x <= 1280
Example:
1024

Response

200

application/json
Successfully generated image

​
id
stringrequired
The ID of the request.

Example:
"generate-image-1234567890"

​
images
string[]required
Base64 encoded image data.

​
timing
objectrequired
Show child attributes

​
request
any
The original request data sent to the API.