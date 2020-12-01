import axios from 'axios';
let URL = 'http://127.0.0.1:5000/api/';
let GetPredictionsURL = URL + 'get_predictions/';

export default {
    axios,
    URL,
    GetPredictionsURL,
}