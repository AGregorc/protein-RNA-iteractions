<template>
  <div id="container-01" class="mol-container"></div>

  <teleport to="#navigation">
    <div class="padding">

      <img alt="Vue logo" src="@/assets/img/logo.png">
      <p> Insert PDB ID </p>
      <label>
        <input list="pdb_list" type="text" v-model="pdb_text">
        <datalist id="pdb_list">
          <option v-for="pdb in all_pdbs" :value="pdb" :key="pdb"/>
        </datalist>

      </label>
      <button @click="load"> Load </button>

      <div v-if="is_model_visible" id="style-panel" >
<!--        <button  @click="changeProteinStyle"> Change style </button>-->
        <select v-model="protein_style_selected">
          <option v-for="style in protein_styles" v-bind:key="style">
            {{ style }}
          </option>
        </select>
        <br>

        <input id="hide-rna" type="checkbox" v-model="rna_hide">
        <label for="hide-rna">Hide RNA</label>
        <br>
        <button @click="recenter">Recenter</button>
        <br>

        <input id="res-labels" type="checkbox" v-model="toggle_res_labels">
        <label for="res-labels">Show residue labels</label>
        <br>
        <br>
        <label for="pred_threshold">Prediction threshold </label><span v-text="curr_threshold"></span>
        <br>
        <input id="pred_threshold" type="range" min="0" max="1" v-model="curr_threshold" step="0.01" />
        <br>
        <button @click="to_optimal">To optimal threshold</button>
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
      is_model_visible: false,
      pdb_text: '',
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
      rna_hide: false,
      toggle_res_labels: false,
      optimal_threshold: 0.5001,
      curr_threshold: 0.5,
    }
  },
  watch: {
    protein_style_selected: function() {
      this.changeProteinStyle();
    },
    rna_hide: function() {
      viewer.setStyle({}, {cartoon: {hidden: this.rna_hide, colorscheme: 'Jmol'}});  /* default style */
      this.changeProteinStyle();
    },
    toggle_res_labels: function () {
      this.resLabels();
    },
    curr_threshold: function () {
      this.changeProteinStyle();
    }
  },
  mounted() {
    let element = document.getElementById('container-01');
    let config = { backgroundColor: 'gray' };
    viewer = window.$3Dmol.createViewer( element, config );
    API.axios.get(API.ListAllPdbsURL)
      .then(response => {
        console.log(response.data.all_pdbs);
        this.all_pdbs = response.data.all_pdbs;
      })
  },
  methods: {
    load() {
      if (!this.all_pdbs.includes(this.pdb_text) && !this.all_pdbs.includes(this.pdb_text+'.pdb')) {
        alert('Sorry but we do not recognize this pdb ' + this.pdb_text);
        return;
      }
      API.axios
        .get(API.GetPredictionsURL + this.pdb_text)
        .then(response => {
          console.log(response.data)
          this.predictions = response.data.predictions;
          this.protein_chains = response.data.protein_chains;
          let file = response.data.file;

          console.log(this.protein_chains);

          let v = viewer;
          v.removeAllModels();
          v.addModel( file, "pdb" );                       /* load data */
          v.setStyle({}, {cartoon: {colorscheme: 'Jmol'}});  /* default style */
          this.changeProteinStyle();
          v.zoomTo();                                      /* set camera */
          v.render();                                      /* render scene */
          v.zoom(1.2, 1000);
          this.is_model_visible = true;
        });
    },
    changeProteinStyle() {
      console.log('change style')
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
  }
}
</script>

<style scoped>

.mol-container {
  width: 100%;
  height: 100%;
  position: relative;
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
</style>