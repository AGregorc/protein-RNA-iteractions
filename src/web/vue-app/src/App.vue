<template>
    <MainVisualization/>
</template>

<script>

import MainVisualization from "@/components/MainVisualization";
export default {
  name: 'App',
  components: {
    MainVisualization
  }
}
</script>

<style>
#app {
  height: 100%;
  width: 100%;
}
/*#app {*/
/*  height: 100vh;*/
/*  width: 100vw;*/

/*  font-family: Avenir, Helvetica, Arial, sans-serif;*/
/*  -webkit-font-smoothing: antialiased;*/
/*  -moz-osx-font-smoothing: grayscale;*/

/*  display: flex;*/
/*  float: left;*/
/*  color: #2c3e50;*/
/*}*/

img {
  width: 8em;
  height: auto;
}
</style>
