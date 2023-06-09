from flask import Flask,request, render_template
from final import fr
#import face_recognition
from werkzeug.utils import secure_filename
from scipy.spatial.distance import cosine
import cv2
import os

UPLOAD_FOLDER = r'static/images/'
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
app = Flask(__name__, template_folder='templates')
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/upload')
def upload_file():
   return render_template('upload.html')
	
@app.route('/kyc', methods = ['POST'])
def upload_files():
   if request.method == 'POST':

        aadhar = request.files['aadhar']
        filename = secure_filename(aadhar.filename)
        filename = str(filename).replace(" ",'')
        if filename.split('.')[-1] in ALLOWED_EXTENSIONS:
            aadhar.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            id_path = r'static/images/'+str(filename)
        else:
            return "ID card file extention not allowed"
        #image_path = fr.my_img()

        image = request.files['image']
        filename = secure_filename(image.filename)
        filename = str(filename).replace(" ",'')
        if filename.split('.')[-1] in ALLOWED_EXTENSIONS:
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image_path = r'static/images/'+str(filename)
        else:
            return "Image file extention not allowed"

        id_verify=fr.id_verification(id_path)
        if id_verify:
            #print(id_verify)
            frame, status1=fr.face_detect(id_path)
            if status1:
                #adhar_picture = face_recognition.load_image_file(id_path)
                my_adhar_encoding = frame
            else:
                os.remove(image_path)
                return 'Upload clear image of id_proof'

            frame, status2=fr.face_detect(image_path)
            if status2:
                #image_picture = face_recognition.load_image_file(image_path)
                my_image_encoding = frame
            else:
                os.remove(image_path)
                os.remove(id_path)
                return 'Please check, Your face should be infront of camera and retry again'

            if status1 & status2:
                model_scores = fr.get_model_scores(my_adhar_encoding, my_image_encoding)
                #face_distance = face_recognition.face_distance([my_adhar_encoding],my_image_encoding)

                if cosine(model_scores[0], model_scores[1]) <= 0.5:
                    os.remove(image_path)
                    os.remove(id_path)
                    return "Congratulations !!! Your KYC is Completed !!! with distance of " + str(cosine(model_scores[0], model_scores[1]) )
                else:
                    os.remove(image_path)
                    os.remove(id_path)
                    return "Soryy !!! KYC Failed ! with distance of " + str(cosine(model_scores[0], model_scores[1]) )
        else:
            os.remove(image_path)
            os.remove(id_path)
            return "You have uploaded wrong id_proof"
        
        return "file upload successful"
    
if __name__ == "__main__":
    
    app.run(host='0.0.0.0', port=8080,debug=True)