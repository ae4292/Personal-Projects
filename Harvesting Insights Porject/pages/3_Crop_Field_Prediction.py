import streamlit as st

st.set_page_config(page_title="Crop Field Prediction")

st.markdown("<h1 style='text-align: center; color: black;'>Crop Field Prediction</h1>", unsafe_allow_html=True)

st.divider()

sat_options = ["Sentinel Hub"]
coordinates_string = st.selectbox("Choose remote sensing option", sat_options)

coord_options = ["-121.428729,39.017344", "-121.429780,39.008294", "-121.124193,37.897960"]
coordinates_string = st.selectbox("Select Longitude/Latitude", coord_options)

# Split the string by comma
coordinates_list = coordinates_string.split(',')

# Convert strings to float
low_left_x = float(coordinates_list[0])
low_left_y = float(coordinates_list[1])
submit_button = st.button("Submit")

if (submit_button):
    import numpy as np
    import tensorflow as tf
    from PIL import Image
    from keras.preprocessing import image
    import tensorflow as tf
#    from tensorflow.keras import losses, Model
    from tensorflow.keras import losses
    from tensorflow.keras.applications.resnet import preprocess_input
    from tensorflow.keras.preprocessing import image
    import cv2
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import Model
    import matplotlib.pyplot as plt
    import ssl

    from sentinelhub import SHConfig
    from sentinelhub import (
        CRS,
        BBox,
        DataCollection,
        DownloadRequest,
        MimeType,
        MosaickingOrder,
        SentinelHubDownloadClient,
        SentinelHubRequest,
        bbox_to_dimensions,
    )

    def setup_sentinel_hub(client_id, client_secret, base_url, token_url):
        config = SHConfig()
        config.sh_client_id = "sh-ca20bb40-6069-487d-8fe5-db95fcb8ddbf"
        config.sh_client_secret = "yDWt2vH6vw8CPnz6oAPF5Oc93JoSXzPh"
        config.sh_base_url = 'https://sh.dataspace.copernicus.eu'
        config.sh_token_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
        if not config.sh_client_id or not config.sh_client_secret:
            print("Warning! To use Process API, please provide the credentials (OAuth client ID and client secret).")
        return config

    def get_image_from_coordinates(config, coordinates, resolution, start_date, end_date):
        coordinates_bbox = BBox(bbox=coordinates, crs=CRS.WGS84)
        coordinates_size = bbox_to_dimensions(coordinates_bbox, resolution=resolution)

        evalscript_true_color = """
            //VERSION=3

            function setup() {
                return {
                    input: [{
                        bands: ["B02", "B03", "B04"]
                    }],
                    output: {
                        bands: 3
                    }
                };
            }

            function evaluatePixel(sample) {
                return [sample.B04, sample.B03, sample.B02];
            }
        """
        request_true_color = SentinelHubRequest(
            evalscript=evalscript_true_color,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L1C.define_from("s2l1c", service_url=config.sh_base_url),
                    time_interval=(start_date, end_date),
                    mosaicking_order=MosaickingOrder.LEAST_CC,
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
            bbox=coordinates_bbox,
            size=coordinates_size,
            config=config,
        )

        true_color_imgs = request_true_color.get_data()

        return true_color_imgs

    def save_image(image_data, output_file, factor=1, clip_range=(0, 1)):
        """
        Save an image from image data to disk.

        Args:
            image_data (numpy.ndarray): The image data.
            output_file (str): The file path where the image will be saved.
            factor (float): A scaling factor applied to the image.
            clip_range (tuple): A tuple specifying the range of pixel values to clip.

        Returns:
            None
        """
        # Apply scaling factor and clipping
        scaled_image_data = np.clip(image_data * factor, *clip_range)

        # Convert to 8-bit unsigned integer
        scaled_image_data = (255 * scaled_image_data).astype(np.uint8)

        # Create an image object from the image data
        image = Image.fromarray(scaled_image_data.squeeze())

        # Save the image to disk
        image.save(output_file)
    ssl._create_default_https_context = ssl._create_unverified_context

    base_model = tf.keras.applications.ResNet152(weights = 'imagenet', include_top=False, input_shape= (282,368,3), classes = 50)
    layer_numbers = len(base_model.layers)
    layers_to_train = 10

    for index, layer in enumerate(base_model.layers):
        if index > (layer_numbers - layers_to_train - 1):
            layer.trainable = True
        else:
            layer.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dropout(.4)(x)
    predictions = tf.keras.layers.Dense(50, activation = 'softmax')(x)

    head_model = Model(inputs = base_model.input, outputs = predictions)
    head_model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])

    head_model.load_weights('models/crop_prediction_trained_model.h5')

    client_id = "sh-ca20bb40-6069-487d-8fe5-db95fcb8ddbf"
    client_secret = "yDWt2vH6vw8CPnz6oAPF5Oc93JoSXzPh"
    base_url = 'https://sh.dataspace.copernicus.eu'
    token_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
    config = setup_sentinel_hub(client_id, client_secret, base_url, token_url)
    start_date = "2023-05-01"
    end_date = "2023-09-30"
    resolution = 10
    image_path = "test.png"

    #-122.622914,38.804868
    coordinates = (low_left_x,low_left_y, low_left_x + 0.005, low_left_y + 0.005)
    image_data = get_image_from_coordinates(config, coordinates, resolution, start_date, end_date)
    save_image(image_data[0], image_path, factor=3 / 255, clip_range=(0, 1))

    # Load the image directly
    img = image.load_img(image_path, target_size=(282, 368))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    st.image(img, caption='Farm Land Satellite Image', use_column_width=True)
    st.write("")

    crop_list = [
    "Alfalfa",
    "Almonds",
    "Apples",
    "Barley",
    "Barren",
    "Cantaloupes",
    "Cherries",
    "Citrus",
    "Corn",
    "Cotton",
    "Dbl_Crop_Oats_Corn",
    "Dbl_Crop_Triticale_Corn",
    "Dbl_Crop_WinWht_Corn",
    "Dbl_Crop_WinWht_Cotton",
    "Dbl_Crop_WinWht_Sorghum",
    "Developed_High_Intensity",
    "Developed_Low_Intensity",
    "Developed_Med_Intensity",
    "Developed_Open_Space",
    "Durum_Wheat",
    "Fallow_Idle_Cropland",
    "Garlic",
    "Grapes",
    "Grassland_Pasture",
    "Herbaceous_Wetlands",
    "Honeydew_Melons",
    "Lettuce",
    "Oats",
    "Olives",
    "Onions",
    "Open_Water",
    "Other_Hay_Non_Alfalfa",
    "Other_Tree_Crops",
    "Peaches",
    "Pears",
    "Peas",
    "Pistachios",
    "Plums",
    "Pomegranates",
    "Prunes",
    "Rice",
    "Shrubland",
    "Sorghum",
    "Spring_Wheat",
    "Tomatoes",
    "Triticale",
    "Walnuts",
    "Watermelons",
    "Winter_Wheat",
    "Woody_Wetlands"
]

    prediction = head_model.predict(img_array)
    predicted_class = np.argmax(prediction)
    st.write(f"Predicted class: {crop_list[predicted_class]}")

    st.divider()

    def generate_grad_cam(model, img_path, class_index, alpha=0.6):
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(282, 368))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.resnet.preprocess_input(img_array)

        last_conv_layer = None
        for layer in reversed(model.layers):
            if 'conv' in layer.name:
                last_conv_layer = layer
                break

        if last_conv_layer is None:
            raise ValueError("No convolutional layer found in the model.")

        #grad_model = Model(inputs=[model.inputs], outputs=[last_conv_layer.output, model.output])
        grad_model = Model(inputs=model.inputs, outputs=[last_conv_layer.output, model.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)[0]
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]

        heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)
        heatmap = np.maximum(heatmap, 0)

        max_value = np.max(heatmap)
        if max_value != 0:
            heatmap /= max_value

        heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[2]))

        original_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_RGB2BGR)
        heatmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

        superimposed_img = cv2.addWeighted(original_img, alpha, heatmap, 1 - alpha, 0)
        superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR)

        return superimposed_img

    class_index = predicted_class
    alpha = 0.4

    superimposed_img = generate_grad_cam(head_model, image_path, class_index, alpha)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    original_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_RGB2BGR)
    plt.imshow(original_img)
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.title('Grad-CAM Result')

    st.pyplot(plt)

    st.markdown('>Gradient Weighted Class Activation Mapping for the selected farmland region. \
                The GRAD-CAM shows areas of the region where the crop is being grown')
