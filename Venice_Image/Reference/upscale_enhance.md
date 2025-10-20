curl --request POST \
  --url https://api.venice.ai/api/v1/image/upscale \
  --header 'Authorization: Bearer <token>' \
  --header 'Content-Type: application/json' \
  --data '{
  "enhance": true,
  "enhanceCreativity": 0.5,
  "enhancePrompt": "gold",
  "image": "iVBORw0KGgoAAAANSUhEUgAAAgAAAAIACAIAAAB7GkOtAAAAIGNIUk0A...",
  "scale": 2
}'

Upscale and Enhance
Upscale or enhance an image based on the supplied parameters. Using a scale of 1 with enhance enabled will only run the enhancer. The image can be provided either as a multipart form-data file upload or as a base64-encoded string in a JSON request.
​
Authorization
stringheaderrequired
Bearer authentication header of the form Bearer <token>, where <token> is your auth token.

Body

application/json
Upscale or enhance an image based on the supplied parameters. Using a scale of 1 with enhance enabled will only run the enhancer.

​
image

any
required
The image to upscale. Can be either a file upload or a base64-encoded string. Image dimensions must be at least 65536 pixels and final dimensions after scaling must not exceed 16777216 pixels.

​
enhance

boolean
default:false
Whether to enhance the image using Venice's image engine during upscaling. Must be true if scale is 1.

Example:
true

​
enhanceCreativity
number | nulldefault:0.5
Higher values let the enhancement AI change the image more. Setting this to 1 effectively creates an entirely new image.

Required range: 0 <= x <= 1
Example:
0.5

​
enhancePrompt
string
The text to image style to apply during prompt enhancement. Does best with short descriptive prompts, like gold, marble or angry, menacing.

Maximum length: 1500
Example:
"gold"

​
replication
number | nulldefault:0.35
How strongly lines and noise in the base image are preserved. Higher values are noisier but less plastic/AI "generated"/hallucinated. Must be between 0 and 1.

Required range: 0 <= x <= 1
Example:
0.35

​
scale
numberdefault:2
The scale factor for upscaling the image. Must be a number between 1 and 4. Scale of 1 requires enhance to be set true and will only run the enhancer. Scale must be > 1 if enhance is false. A scale of 4 with large images will result in the scale being dynamically set to ensure the final image stays within the maximum size limits.

Required range: 1 <= x <= 4
Example:
2

Response

200

image/png
OK

The response is of type file.