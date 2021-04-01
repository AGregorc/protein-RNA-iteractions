import axios from 'axios';
let URL = process.env.API_URL || 'http://' + window.location.hostname + ':5004/api/';

let GetPredictionsURL = URL + 'get_predictions/';
let ListAllPdbsURL = URL + 'list_all_pdbs';

console.log("API URL is " + URL);

export default {
    axios,
    URL,
    GetPredictionsURL,
    ListAllPdbsURL,
}