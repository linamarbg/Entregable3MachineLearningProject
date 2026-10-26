// Desplaza a un ancla de la URL (/#formulacion, /eda#fuga) cuando Dash dibuja la página: el navegador busca el ancla al
// cargar, antes de que exista el elemento, y los gráficos que llegan después cambian las alturas. Durante unos segundos
// se mantiene el elemento en su lugar mientras la página termina de dibujarse, salvo que la persona empiece a desplazarse.
(function () {
  var reloj = null;
  function detener() { if (reloj) { clearInterval(reloj); reloj = null; } }
  ["wheel", "touchstart", "keydown", "mousedown"].forEach(function (ev) {
    window.addEventListener(ev, detener, { passive: true });
  });
  function irAlAncla() {
    var id = decodeURIComponent(window.location.hash.slice(1));
    if (!id) return;
    detener();
    var inicio = Date.now(), visto = null;
    reloj = setInterval(function () {
      var el = document.getElementById(id);
      if (el) {
        visto = visto || Date.now();
        var margen = parseFloat(getComputedStyle(el).scrollMarginTop) || 0;
        var destino = el.getBoundingClientRect().top + window.scrollY - margen;
        // «instant»: Bootstrap pone scroll-behavior: smooth y reiniciar la animación en cada paso la dejaría quieta
        if (Math.abs(window.scrollY - destino) > 4) window.scrollTo({ top: destino, behavior: "instant" });
      }
      if ((visto && Date.now() - visto > 6000) || Date.now() - inicio > 20000) detener();
    }, 200);
  }
  irAlAncla();
  window.addEventListener("hashchange", irAlAncla);

  // En pantallas angostas la barra de pestañas se desplaza de lado: al abrir /modelos?tab=... se muestra la activa.
  var intentosPestana = 0;
  var relojPestana = setInterval(function () {
    var barra = document.querySelector(".nav-tabs");
    var activa = barra && barra.querySelector(".nav-link.active");
    if (activa) {
      clearInterval(relojPestana);
      if (barra.scrollWidth > barra.clientWidth) barra.scrollLeft = activa.offsetLeft - 16;
    } else if (++intentosPestana > 60) {
      clearInterval(relojPestana);
    }
  }, 200);
})();
