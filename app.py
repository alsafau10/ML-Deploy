import os
from ultralytics import YOLO
from PIL import Image
from datetime import datetime
from flask_cors import CORS
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
model = YOLO('best.pt')


UPLOAD_FOLDER = 'static/upload/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSION = {'png','jpg','jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSION

#rute
@app.route('/',methods=['GET','POST'])
def main():
    return render_template('index.html')
    
@app.route('/submit',methods=['POST'])
def predict():
    if 'file' not in request.files:
        resp = jsonify({'message': 'No image in the request'})
        resp.status_code = 400
        return resp
    files = request.files.getlist('file')
    filename= 'temp_image.png'
    errors = {}
    success = False
    for file in files:
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            success = True
        else:
            errors['message']= 'file type of {} is not allowed'.format(file.filename)

    if not success:
        resp = jsonify(errors)
        resp.status_code = 400
        return resp
    img_url = os.path.join(app.config['UPLOAD_FOLDER'],filename)

    img = Image.open(img_url)
    now = datetime.now()
    timeName =  now.strftime("%d%m%y-%H%M%S")+'.png'
    predict_image_path = 'static/upload/'+ timeName
    image_predict = predict_image_path
    img.save(image_predict,format='png')

    img = predict_image_path
    print(timeName)

    save_dir = 'static/image/result'
    os.makedirs(save_dir, exist_ok=True)
    results = model.predict(img,save=True, imgsz=736,project =save_dir ,exist_ok=True)
    metadata_img= results[0].tojson()
    return render_template('index.html',img_path = timeName, results= metadata_img)


if __name__ == '__main__':
    app.run(debug=True)


