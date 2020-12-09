import axios from 'axios';
let URL = 'http://' + window.location.hostname + ':5000/api/';
let GetPredictionsURL = URL + 'get_predictions/';

export default {
    axios,
    URL,
    GetPredictionsURL,
}