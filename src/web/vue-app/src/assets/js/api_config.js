import axios from 'axios';
let URL = 'http://' + window.location.hostname + ':5000/api/';
let GetPredictionsURL = URL + 'get_predictions/';
let ListAllPdbsURL = URL + 'list_all_pdbs';

export default {
    axios,
    URL,
    GetPredictionsURL,
    ListAllPdbsURL,
}