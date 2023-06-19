import numpy as np
import keras
from datetime import date
import pickle
from flask_mysqldb import MySQL, MySQLdb
from flask import Flask, jsonify, make_response,request,flash,redirect,render_template, session,url_for, send_from_directory
from itsdangerous import json
from werkzeug.utils import secure_filename
import os
import re
import random
import string
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model



#-----------Konfigurasi------------

app = Flask(__name__)
app.secret_key = "xxxxx"

#run_with_ngrok(app)
#Konfigurasi folder menyiman upload
UPLOAD_FOLDER_IMG = 'aset/foto_burung'
FOLDER_ICON = 'aset/icon_burung'
FOLDER_SUARA = 'aset/audio_burung'

ALLOWED_EXTENSIONS_IMG = set(['png', 'jpg', 'jpeg','jfif'])
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER_IMG'] = UPLOAD_FOLDER_IMG
app.config["CLIENT_IMAGES"] = FOLDER_ICON
app.config["CLIENT_AUDIO"] = FOLDER_SUARA

#konfigurasi database
app.config['MYSQL_HOST'] = 'localhost' #http://103.235.74.136:888/
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '' #7bb1245644d4f350
app.config['MYSQL_DB'] = 'mibird'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
mysql = MySQL(app)

MODEL_IMG_PATH = 'modelResnet-40+.h5'
model_img = load_model(MODEL_IMG_PATH,compile=False)

pickle_img = open('num_7class_bird.pkl','rb')
num_classes_img = pickle.load(pickle_img)

lemmatizer = WordNetLemmatizer()
nltk.download('omw-1.4')
model_chatbot = load_model("chatbot\chatbot_model.h5")
intents = json.loads(open("chatbot\intents.json").read())
words = pickle.load(open("chatbot\words.pkl", "rb"))
classes = pickle.load(open("chatbot\classes.pkl", "rb"))

def allowed_file_img(filename):     
  return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_IMG

# API
@app.route('/api/predict', methods=['POST'])
def predict():
    
    file = request.files['file']
    if file.filename == '':
      return jsonify({
            "pesan":"tidak ada file image yang dipilih"
          })
    if file and allowed_file_img(file.filename):
      
        letters = string.ascii_lowercase
        result_str = ''.join(random.choice(letters) for i in range(5))
        
        filename = secure_filename(result_str+".jpg")
        file.save(os.path.join(app.config['UPLOAD_FOLDER_IMG'], filename))
        path=("aset/foto_burung/"+filename)
    
        today = date.today()
        cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cur.execute("INSERT INTO riwayat (nama_file,path, prediksi,akurasi,tanggal) VALUES (%s,%s,%s,%s,%s)", (filename,path,'No predict',int(0), today.strftime("%d/%m/%Y")))
        mysql.connection.commit()

        img=keras.utils.load_img(path,target_size=(224,224))
        img1=keras.utils.img_to_array(img)
        img1=img1/255
        img1=np.expand_dims(img1,[0])
        predict=model_img.predict(img1)
        classes=np.argmax(predict,axis=1)
        
        for key,values in num_classes_img.items():
            if classes==values:
                accuracy = float(round(np.max(model_img.predict(img1))*100,2))
                cur.execute("SELECT * FROM data_burung WHERE nama_burung=%s",(str(key),))
                result = cur.fetchone()
                cur.execute("""
                    UPDATE riwayat
                    SET prediksi = %s,
                        akurasi = %s
                    WHERE nama_file = %s
                """, (str(key),accuracy,filename))
                mysql.connection.commit()
            
                print("The predicted image of the bird is: "+str(key)+" with a probability of "+str(accuracy)+"%")            
    
                if accuracy>=65:
                    return jsonify({
                    "Nama_Burung":str(key),
                    "Accuracy":str(accuracy)+"%",
                    "Spesies" : result['spesies'],
                    "Makanan" : result['makanan'],
                    "Anatomi" :  result['anatomi'],
                    "Habitat" :  result['habitat'],
                    "Umur" :  result['umur'],
                    "Url_image" :  result['gambar'],
                    "Url_suara" :  result['suara'],
                    "Url_video" :  result['video']
                    }) 
                elif accuracy <65 :
                    return jsonify({
                    "Nama_Burung":"Burung Belum dikenal \n /Foto bukan burung",
                    "Accuracy":str(accuracy)+"%",
                    "Spesies" : "",
                    "Makanan" : "",
                    "Anatomi" :  "",
                    "Habitat" :  "",
                    "Umur" :  "",
                    "Url_image" :  "",
                    "Url_suara" :  "",
                    "Url_video" :  ""
                })       

@app.route('/api/dataBurung', methods=['GET'])
def api_dataBurung():
    
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)    
    cur.execute("SELECT * FROM data_burung ")
    result = cur.fetchall()
    return make_response(jsonify({"data_burung":[dict(row) for row in result]}), 200)

@app.route('/api/saran', methods=['POST'])
def ApiSaran():
    saran = request.form.get('saran')
    today = date.today()
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("INSERT INTO saran (saran_burung,tanggal) VALUES (%s,%s)", (saran, today.strftime("%d/%m/%Y")))
    mysql.connection.commit()
    return jsonify({"ResponseSaran":True})

@app.route("/api/chatbot", methods=["POST"])
def chatbot_response():
    msg = request.form["msg"]
    ints = predict_class(msg, model_chatbot)
    res = getResponse(ints, intents)
    return jsonify({"response_chatbot":res})

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result

# Get Image file Routing
@app.route("/get-image/<path:image_name>",methods = ['GET','POST'])
def get_image(image_name):
    try:
        return send_from_directory(app.config["CLIENT_IMAGES"], filename=image_name)
    except FileNotFoundError:
        abort(404)

# Get audio file Routing
@app.route("/get-audio/<path:audio_name>",methods = ['GET','POST'])
def get_audio(audio_name):

    try:
        return send_from_directory(app.config["CLIENT_AUDIO"], filename=audio_name)
    except FileNotFoundError:
        abort(404)

# Admin
@app.route('/admin')
def admin():
    return render_template("login.html")
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        curl = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        curl.execute("SELECT * FROM admin WHERE username=%s", (username,) )
        user = curl.fetchone()
        curl.close()

        if user is not None and len(user) > 0:
            if password == user['password'] and username == user['username']:
                session['logged_in'] = True
                
                return redirect(url_for('dataBurung'))
            else:
                error = "Gagal, username dan password tidak cocok"
                return render_template('login.html', error=error)
        else:
            error = "Gagal, user tidak ditemukan"
            return render_template('login.html', error=error)
    else:
        return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/dataBurung')
def dataBurung():
    error = None
    if session.get('logged_in') :
        cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cur.execute('SELECT * FROM data_burung')
        data = cur.fetchall()
        cur.close()
        return render_template('dataBurung.html',dataBurung  = data)
    else :
        error = "Anda Belum Login"
        return render_template('login.html', error=error)
        
@app.route('/tambahData')
def tambahData():
    return render_template('tambahData.html')

@app.route('/daftarBurung', methods=["POST"])
def daftarBurung():
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    if request.method == "POST":
        nm_burung = request.form['nm_burung']
        spesies = request.form['spesies']
        makanan = request.form['makanan']
        anatomi = request.form['anatomi']
        habitat = request.form['habitat']
        umur = request.form['umur']
        gambar = request.form['gambar']
        suara = request.form['suara']
        video = request.form['video']
        if not re.match(r'[A-Za-z]+', nm_burung):
            flash("Nama harus pakai huruf Dong!")
        
        else:
            cur.execute("INSERT INTO data_burung (nama_burung,spesies,makanan,anatomi,habitat,umur,gambar,suara,video) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)", (nm_burung,spesies, makanan, anatomi, habitat, umur, gambar, suara, video))
            mysql.connection.commit()
            flash('Data Burung berhasil ditambah')
            return redirect(url_for('dataBurung'))

    return render_template("tambahData.html")

@app.route('/editBurung/<nama_burung>', methods = ['POST', 'GET'])
def editBurung(nama_burung):
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    
    cur.execute('SELECT * FROM data_burung WHERE nama_burung = %s', [nama_burung])
    data = cur.fetchone()
    cur.close()
    return render_template('editBurung.html', editBurung = data)

@app.route('/updateBurung/<nama_burung>', methods=['POST'])
def updatBurung(nama_burung):
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    if request.method == 'POST':
        spesies = request.form['spesies']
        makanan = request.form['makanan']
        anatomi = request.form['anatomi']
        habitat = request.form['habitat']
        umur = request.form['umur']
        gambar = request.form['gambar']
        suara = request.form['suara']
        video = request.form['video']
        if not re.match(r'[A-Za-z]+', nama_burung):
            flash("Nama harus pakai huruf Dong!")
        else:
          cur.execute("""
              UPDATE data_burung
              SET spesies = %s,
                  makanan = %s,
                  anatomi = %s,
                  habitat = %s,
                  umur = %s,
                  gambar = %s,
                  suara = %s,
                  video = %s
              WHERE nama_burung = %s
            """, (spesies, makanan, anatomi, habitat, umur, gambar, suara, video, nama_burung))
          flash('Data Burung berhasil diupdate')
          mysql.connection.commit()
          cur.close()
          return render_template("popUpEdit.html")

    return render_template("dataBurung.html")

@app.route('/riwayat')
def riwayat():
    if session.get('logged_in') :
        cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cur.execute("SELECT * FROM riwayat")
        dataRiwayat = cur.fetchall()
        cur.close()
        return render_template('riwayat.html',riwayat  = dataRiwayat)
    else :
        error = "Anda Belum Login"
        return render_template('login.html', error=error)
    
@app.route('/hapusRiwayat/<nama_file>', methods = ['POST','GET'])
def hapusRiwayat(nama_file):
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
  
    cur.execute('DELETE FROM riwayat WHERE nama_file =%s',[nama_file])
    mysql.connection.commit() 
    flash(' Berhasil Dihapus!')
    return redirect(url_for('riwayat'))

@app.route('/saran')
def saran():
    if session.get('logged_in') :
        cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cur.execute("SELECT * FROM saran")
        dataSaran = cur.fetchall()
        cur.close()
        return render_template('saran.html',saran  = dataSaran)
    else :
        error = "Anda Belum Login"
        return render_template('login.html', error=error)
    
@app.route('/hapusSaran/<saran_burung>', methods = ['POST','GET'])
def hapusSaran(saran_burung):
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
  
    cur.execute('DELETE FROM saran WHERE saran_burung =%s',[saran_burung])
    mysql.connection.commit() 
    flash(' Berhasil Dihapus!')
    return redirect(url_for('saran'))


if __name__ == '__main__':

  app.run(debug=True, host="0.0.0.0",)
