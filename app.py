import io
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from flask import Flask, render_template, Response, request

# Import functions from the utils module
from utils import update_buffer, get_last_data, compute_band_powers, julia, create_custom_colormap
from pylsl import StreamInlet, resolve_byprop

app = Flask(__name__)

BUFFER_LENGTH = 5
EPOCH_LENGTH = 1
OVERLAP_LENGTH = 0
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH
INDEX_CHANNEL = [0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/plot')
def plot():

    def generate_plot():
        
        # Search for active LSL streams
        print('Looking for an EEG stream...')
        streams = resolve_byprop('type', 'EEG', timeout=2)
        if len(streams) == 0:
            raise RuntimeError('Can\'t find EEG stream.')
        else:
            print('Found it!')
            print(streams)
            
        # Set active EEG stream to inlet and apply time correction
        print("Start acquiring data")
        inlet = StreamInlet(streams[0], max_chunklen=12)
        eeg_time_correction = inlet.time_correction()

        # Get the stream info
        info = inlet.info()
        fs = int(info.nominal_srate())

        # Initialize raw EEG data buffer
        eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
        filter_state = None  # for use with the notch filter

        # Compute the number of epochs in "buffer_length"
        n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) /
                                    SHIFT_LENGTH + 1))

        # Initialize the band power buffer (for plotting)
        # bands will be ordered: [delta, theta, alpha, beta]
        band_buffer = np.zeros((n_win_test, 4))

        print('Press Ctrl-C in the console to break the while loop.')
    
        fig, ax = plt.subplots()

        while True:
            # Obtain EEG data from the LSL stream
            eeg_data, timestamp = inlet.pull_chunk(
                timeout=1, max_samples=int(SHIFT_LENGTH * fs))

            # Only keep the channel we're interested in
            ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]

            # Update EEG buffer with the new data
            eeg_buffer, filter_state = update_buffer(
                eeg_buffer, ch_data, notch=True,
                filter_state=filter_state)

            # Get newest samples from the buffer
            data_epoch = get_last_data(eeg_buffer,
                                                EPOCH_LENGTH * fs)

            # Compute band powers
            band_powers = compute_band_powers(data_epoch, fs)
            band_buffer, _ = update_buffer(band_buffer,
                                                    np.asarray([band_powers]))

            delta = band_powers[0]
            theta = band_powers[1]
            alpha = band_powers[2]
            beta = band_powers[3]

            # Create a Julia fractal with a complex constant based on the EEG values
            c = complex(alpha, beta)
            julia_img = julia(c)

            # Determine the color based on EEG values using the custom colormap
            custom_cmap = create_custom_colormap()
            color_value = (theta + delta) / 2
            
            # Rescaling down to [0.7, 3]
            max_voltage = 3
            min_voltage = 0.7
            
            color_value = max(max(color_value, min_voltage), min(max_voltage, color_value))
            rescaled_color_value = (color_value - min_voltage) / (max_voltage - min_voltage)
            rescaled_color_value = max(0.0, min(1.0, rescaled_color_value))
            
            color = custom_cmap(rescaled_color_value)

            # Create an RGB image with the same dimensions as the grayscale Julia image
            color_image = np.zeros((julia_img.shape[0], julia_img.shape[1], 3), dtype=np.uint8)

            # Set the color in each pixel of the RGB image
            color_image[:, :, 0] = color[0] * 255
            color_image[:, :, 1] = color[1] * 255
            color_image[:, :, 2] = color[2] * 255

            # Display the Julia fractal with the determined color
            img_data = io.BytesIO()
            ax.imshow(julia_img, cmap='gray')
            ax.imshow(color_image, alpha=0.5)
            ax.set_title(f'Alpha: {alpha:.2f}, Beta: {beta:.2f}, Theta: {theta:.2f}, Delta: {delta:.2f}')
            
            fig.savefig(img_data, format='png', bbox_inches='tight', pad_inches=0, transparent=True)

            img_data.seek(0)
            yield (b'--frame\r\n'
                b'Content-Type: image/png\r\n\r\n' + img_data.read() + b'\r\n')


    return Response(generate_plot(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
