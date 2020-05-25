import flask
import Regression

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/regres/<filename>', methods=['GET'])
def regres(filename):
    serv = Regression.Regres(filename)
    result = serv.predictValue(0);
    resultString = str(result).strip('[]')

    return resultString

app.run()