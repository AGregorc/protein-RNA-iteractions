<template>
  <div id="container-01" class="mol-container"></div>

  <teleport to="#navigation">
    <div class="padding">

      <img alt="Vue logo" id="logo" src="@/assets/img/logo.png">

      <form @submit.prevent>
        <div class="mb-3">
          <label for="pdb_id" class="form-label"> Insert PDB ID </label>
          <input type="text" v-model="pdb_text" id="pdb_id" class="form-control">
        </div>
        <div class="mb-3">
          <label for="models_list" class="form-label"> Selected model </label>
          <select v-model="selected_model" id="models_list" class="form-select">
            <option v-for="model in models_list" :key="model">
              {{ model }}
            </option>
          </select>
        </div>
        <button @click="load" class="btn btn-dark"> Load </button>

        <div v-if="is_loading"  class="ms-4" id="loading-spinner">
          <div class="spinner-border text-light" role="status">
            <span class="visually-hidden">Loading...</span>
          </div>
        </div>
      </form>


      <div v-if="is_model_visible" id="style-panel" >
        <label for="protein_style"> </label>
        <select v-model="protein_style_selected" class="form-select form-select-sm width-auto" id="protein_style">
          <option v-for="style in protein_styles" v-bind:key="style">
            {{ style }}
          </option>
        </select>
        <input id="show-only-proteins" type="checkbox" class="form-check-input" v-model="toggle_aa_only">
        <label for="show-only-proteins" class="ps-1 form-check-label">Show only proteins</label>
        <br>
        <input id="res-labels" type="checkbox" class="form-check-input" v-model="toggle_res_labels">
        <label for="res-labels" class="ps-1 form-check-label">Show residue labels</label>
        <br>
        <button @click="recenter" class="btn btn-light btn-sm">Recenter</button>

        <br>
        <br>
        <label for="pred_threshold" class="">Prediction threshold &nbsp; </label>
        <span v-text="curr_th_text" ></span>
        <br>
        <input id="pred_threshold" type="range"  min="0" max="1" v-model="curr_threshold" step="0.01"/>
        <br>
        <button @click="to_optimal" class="btn btn-light btn-sm">To optimal threshold</button>
        <br>
      </div>
    </div>
  </teleport>
</template>

<script>
import API from "@/assets/js/api_config";

let viewer = undefined;

export default {
  name: "MainVisualization",
  data: function () {
    return {
      all_pdbs: [],
      models_list: [],
      is_model_visible: false,
      pdb_text: '',
      selected_model: '',
      predictions: undefined,
      protein_chains: [],
      protein_styles: [
        'stick',
        'sphere',
        'line',
        'cross',
        'cartoon',
      ],
      protein_style_selected: 'sphere',
      toggle_aa_only: false,
      toggle_res_labels: false,
      optimal_threshold: 0.5,
      curr_threshold: undefined,
      curr_th_text: '',
      is_loading: false,
    }
  },
  watch: {
    protein_style_selected: function() {
      this.changeProteinStyle();
    },
    toggle_aa_only: function() {
      viewer.setStyle({}, {cartoon: {hidden: this.toggle_aa_only, colorscheme: 'Jmol'}});  /* default style */
      this.changeProteinStyle();
    },
    toggle_res_labels: function () {
      this.resLabels();
    },
    curr_threshold: function () {
      if (this.curr_threshold === undefined)
        this.curr_th_text = '';
      else
        this.curr_th_text = parseFloat(this.curr_threshold).toFixed(2);

      this.changeProteinStyle();
    }
  },
  mounted() {
    let element = document.getElementById('container-01');
    let config = { backgroundColor: 'gray' };
    viewer = window.$3Dmol.createViewer( element, config );
    this.is_loading = true;
    API.axios.get(API.ListAllModelsURL)
      .then(response => {
        // console.log(response.data.all_pdbs);
        this.models_list = response.data.models;
        this.selected_model = this.models_list[0]
        this.is_loading = false;
      })
      .catch(this.errorProcess);
  },
  methods: {
    resetValues() {
      this.pdb_text = ''
      this.is_model_visible = false;
      this.predicitons = undefined;
      this.curr_threshold = undefined;
      this.curr_th_text = '';
      this.is_loading = false;
    },
    load() {
      this.is_loading = true;
      API.axios
        .get(API.GetPredictionsURL + this.pdb_text, {
          params: {
            model: this.selected_model
          }
        })
        .then(response => {
          // console.log(response.data)
          if (!response.data.success) {
            let pdb_text = this.pdb_text;
            this.resetValues();
            setTimeout(() => {
              alert('PDB ID ' + pdb_text + ' does not exist in our database.');
            }, 200);
            return;
          }
          this.predictions = response.data.predictions;
          this.protein_chains = response.data.protein_chains;
          this.optimal_threshold = response.data.optimal_threshold;
          if (this.curr_threshold === undefined)
            this.curr_threshold = this.optimal_threshold;

          let file = response.data.file;

          let v = viewer;
          v.removeAllModels();
          v.addModel( file, "pdb" );                       /* load data */
          v.setStyle({}, {cartoon: {hidden: this.toggle_aa_only, colorscheme: 'Jmol'}});
          this.changeProteinStyle();
          v.zoomTo();                                      /* set camera */
          v.render();                                      /* render scene */
          this.is_model_visible = true;
          this.is_loading = false;
        })
        .catch(this.errorProcess);
    },
    changeProteinStyle() {
      // console.log('change style')
      if (this.protein_chains === undefined)
        return;

      this.protein_chains.forEach(c => {
        let style = {}
        style[this.protein_style_selected] = {colorfunc: this.isInInteraction};
        viewer.setStyle({chain: c}, style);  /* style all atoms */
      })
      viewer.render();
    },
    isInInteraction(atom) {
      // console.log(atom);
      return this.predictions[atom.serial] > this.curr_threshold ? 'red' : 'white';
    },
    resLabels() {
      if (this.toggle_res_labels) {
        this.protein_chains.forEach(c => {
          viewer.addResLabels({ chain : c });
        })
      } else {
        viewer.removeAllLabels();
      }
    },
    recenter() {
      viewer.zoomTo();
    },
    to_optimal() {
      this.curr_threshold = this.optimal_threshold;
    },
    restrict_decimal () {
      this.contract=this.contract.match(/^\d+\.?\d{0,2}/);
    },
    errorProcess(error) {
      this.resetValues();
      alert('Something got wrong when connecting to the server.');
      console.log(error);
    }
  }
}
</script>

<style scoped>

  #logo {
    /*width: 75%;*/
    /*height: auto;*/
    /*margin-bottom: 2em;*/
  }

.mol-container {
  width: 100%;
  height: 100%;
  position: relative;
}

.width-auto {
  width: auto;
}

.padding {
  padding: 1em;
}

#style-panel {
  display: block;
  margin-top: 1em;
  border-top: 5px solid #444444;
  padding-top: 1em;
}

#loading-spinner {
  display: inline-block;
  vertical-align: middle;
}

</style>