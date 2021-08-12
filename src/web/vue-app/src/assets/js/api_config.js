import axios from 'axios';
let URL = process.env.VUE_APP_API_URL || 'http://' + window.location.hostname + ':5004/api/';

let GetPredictionsURL = URL + 'get_predictions/';
let ListAllPdbsURL = URL + 'list_all_pdbs';
let ListAllModelsURL = URL + 'list_models'

console.log("API URL is " + URL);

export default {
    axios,
    URL,
    GetPredictionsURL,
    ListAllPdbsURL,
    ListAllModelsURL,
}