<template>
  <div id="container-01" class="mol-container"></div>

  <teleport to="#navigation">
    <div class="padding">

      <img alt="Vue logo" src="@/assets/img/logo.png">
      <p> Insert pdb file </p>
      <label>
        <input type="text" v-model="pdb_text">

      </label>
      <button @click="load"> Load </button>
      <button v-if="is_model_visible" @click="changeStyle"> Change style </button>
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
      predictions: undefined,
      protein_chains: [],
      pdb_text: '',
      is_model_visible: false,
    }
  },
  mounted() {

    let element = document.getElementById('container-01');

    let config = { backgroundColor: 'gray' };
    viewer = window.$3Dmol.createViewer( element, config );
  },
  methods: {
    load() {
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
          this.protein_chains.forEach(c => {
            v.setStyle({chain: c}, {sphere: {colorfunc: this.isInInteraction}});  /* style all atoms */
          })
          v.zoomTo();                                      /* set camera */
          v.render();                                      /* render scene */
          v.zoom(1.2, 1000);
          this.is_model_visible = true;
        });
    },
    changeStyle() {
      console.log('change style')
      this.protein_chains.forEach(c => {
        viewer.setStyle({chain: c}, {stick: {colorfunc: this.isInInteraction}});  /* style all atoms */
      })
      viewer.render();
    },
    isInInteraction(atom) {
      // console.log(atom);
      return this.predictions[atom.serial] === 1 ? 'red' : 'white';
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
</style>