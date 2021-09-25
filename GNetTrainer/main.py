from flask import Flask, render_template,request,jsonify
from GNetTrainer.com_in_ineuron_ai_utils.utils import decodeImage
from flask_cors import CORS, cross_origin
from GNetTrainer.predict import Predict
import yaml
import GNetTrainer.Training as Training
import webbrowser
from threading import Timer

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"    
        self.classifier = Predict(self.filename)

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
@cross_origin()
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST', 'GET'])
@cross_origin()
def train_function():
    if request.method == 'POST':
        try:
            # Inputs based to data
            TRAIN_DIR = request.form['TRAIN_DIR']
            VAL_DIR = request.form['VAL_DIR']
            DIMENSION = request.form['DIMENSION']
            BATCH_SIZE  = int(request.form['BATCH_SIZE'])
            AUGMENTATION =  bool(request.form['AUGMENTATION'])

            # Model based Data
            MODEL_OBJ =  request.form['MODEL_OBJ']
            MODEL_NAME = request.form['MODEL_NAME']
            EPOCHS = int(request.form['EPOCHS'])
            OPTIMIZER = request.form['OPTIMIZER']
            LOSS_FUNC = request.form['LOSS_FUNC']
            FREEZE_ALL = bool(request.form['FREEZE_ALL'])
            TENSORBOARD = bool(request.form['TENSORBOARD'])
            
            # LR Scheduler based Data
            SCHEDULER = bool(request.form['SCHEDULER'])
            MONITOR = request.form['MONITOR']
            PATIENCE = int(request.form['PATIENCE'])
            FACTOR = float(request.form['FACTOR'])

            # Call function
            dict_file = {
                "train_dir" : TRAIN_DIR,
                "val_dir" : VAL_DIR,
                "dimension" : DIMENSION,
                "batch_size" : BATCH_SIZE,
                "augmentation" : AUGMENTATION,
                "MODEL_OBJ" : MODEL_OBJ,
                "MODEL_NAME": MODEL_NAME,
                "EPOCHS": EPOCHS, 
                "OPTIMIZER": OPTIMIZER,
                "LOSS_FUNC": LOSS_FUNC,
                "FREEZE_ALL": FREEZE_ALL,
                "TENSORBOARD": TENSORBOARD,
                "SCHEDULER": SCHEDULER,
                "MONITOR": MONITOR,
                "PATIENCE": PATIENCE,
                "FACTOR": FACTOR
            }
            print('Dumping Yaml')

            with open('config.yaml', 'w') as file:
                yaml.dump(dict_file, file)
                
            train = Training.train()

            return render_template('train.html', output = train)
 
        except Exception as e:
            print("Input format not proper", end= '')
            print(e)
    else:
        return render_template('train.html')

@app.route('/test',methods=['GET']) 
def predcit():
    return render_template("predict.html")


@app.route("/predict", methods=['POST'])
def predictRoute():
    CLapp = ClientApp()
    image = request.json['image']
    decodeImage(image, CLapp.filename)
    result = CLapp.classifier.predict_image()
    return jsonify(result)


def open_browser():
    webbrowser.open_new('http://127.0.0.1:8080/')


def start_app():
    Timer(1, open_browser).start()
    app.run(host="127.0.0.1", port=8080,debug=True)


if __name__ == "__main__":
    start_app()
