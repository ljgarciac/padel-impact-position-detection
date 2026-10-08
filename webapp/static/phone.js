/* El celular como camara: getUserMedia -> canvas -> JPEG -> relay HTTPS de la computadora.
 *
 * Transporte: WebSocket seguro (wss) y, si no abre (algunas versiones de Safari no lo permiten
 * con un certificado autofirmado aunque la pagina si cargo), POST por cada frame al mismo origen.
 * En ambos casos hay contrapresion: no se manda un frame nuevo mientras el anterior no salio, asi
 * la red nunca acumula frames viejos (la computadora siempre ve lo mas reciente). */
(function () {
  "use strict";

  var TARGET_FPS = 20;        // debe coincidir con PHONE_NOMINAL_FPS de webapp/capture.py
  var JPEG_QUALITY = 0.75;
  var MAX_BUFFERED = 256 * 1024;
  var token = document.body.dataset.token;
  var $ = function (id) { return document.getElementById(id); };

  var stream = null, ws = null, usePost = false, timer = null, wakeLock = null;
  var canvas = document.createElement("canvas");
  var ctx = canvas.getContext("2d");
  var sending = false, sentTimes = [];

  function setState(text, kind) { var t = $("estado-tag"); t.textContent = text; t.className = "tag tag--" + kind; }
  function warn(msg) { $("aviso").hidden = !msg; $("aviso").textContent = msg || ""; }

  async function startCamera() {
    var maxW = parseInt($("calidad").value, 10);
    var constraints = { audio: false, video: {
      facingMode: { ideal: $("lado").value }, width: { ideal: maxW }, height: { ideal: Math.round(maxW * 9 / 16) },
      frameRate: { ideal: 30 } } };
    try {
      stream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (e) {
      stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: true });
    }
    var video = $("previa");
    video.srcObject = stream;
    await video.play();
    $("vacio").hidden = true;
  }

  function stopCamera() {
    if (stream) stream.getTracks().forEach(function (t) { t.stop(); });
    stream = null;
    $("previa").srcObject = null;
    $("vacio").hidden = false;
  }

  function openSocket() {
    return new Promise(function (resolve) {
      var done = false;
      var sock = new WebSocket("wss://" + location.host + "/" + token + "/ws");
      sock.binaryType = "arraybuffer";
      var giveUp = setTimeout(function () { if (!done) { done = true; try { sock.close(); } catch (e) {} resolve(null); } }, 5000);
      sock.onopen = function () { if (!done) { done = true; clearTimeout(giveUp); resolve(sock); } };
      sock.onerror = function () { if (!done) { done = true; clearTimeout(giveUp); resolve(null); } };
      sock.onclose = function (ev) {
        if (done && ws === sock) {
          ws = null;
          if (ev.code === 1008) { warn("Este enlace ya no sirve. Escanea el QR nuevo en la computadora."); stop(); }
          else if (timer) { reconnect(); }
        }
      };
    });
  }

  async function reconnect() {
    setState("Reconectando", "idle");
    ws = await openSocket();
    if (!ws && timer) usePost = true;
    if (timer) setState("Transmitiendo", "run");
    $("via").textContent = ws ? "WebSocket" : "HTTPS (POST)";
  }

  function frameSize() {
    var v = $("previa");
    var maxW = parseInt($("calidad").value, 10);
    var w = v.videoWidth, h = v.videoHeight;
    if (!w || !h) return null;
    if (w > maxW) { h = Math.round(h * maxW / w); w = maxW; }
    return [w, h];
  }

  function tick() {
    if (sending || !stream) return;
    if (ws && ws.bufferedAmount > MAX_BUFFERED) return;  // la red va lenta: se salta este frame
    var size = frameSize();
    if (!size) return;
    if (canvas.width !== size[0] || canvas.height !== size[1]) {
      canvas.width = size[0]; canvas.height = size[1];
      $("res").textContent = size[0] + "×" + size[1];
      if (ws && ws.readyState === 1) ws.send("size " + size[0] + " " + size[1]);
    }
    ctx.drawImage($("previa"), 0, 0, size[0], size[1]);
    sending = true;
    canvas.toBlob(function (blob) {
      if (!blob) { sending = false; return; }
      var sent = function () {
        sending = false;
        var now = performance.now();
        sentTimes.push(now);
        while (sentTimes.length && now - sentTimes[0] > 1000) sentTimes.shift();
        $("fps").textContent = sentTimes.length + " FPS";
      };
      if (ws && ws.readyState === 1) {
        ws.send(blob);
        sent();
      } else if (usePost) {
        fetch("/" + token + "/frame", { method: "POST", body: blob, headers: { "Content-Type": "image/jpeg" } })
          .then(function (r) { if (r.status === 404) { warn("Este enlace ya no sirve. Escanea el QR nuevo."); stop(); } })
          .catch(function () { setState("Sin conexión", "err"); })
          .finally(sent);
      } else {
        sending = false;
      }
    }, "image/jpeg", JPEG_QUALITY);
  }

  async function keepAwake() {
    try { if ("wakeLock" in navigator) wakeLock = await navigator.wakeLock.request("screen"); } catch (e) { /* opcional */ }
  }

  async function start() {
    warn("");
    if (!window.isSecureContext || !navigator.mediaDevices) {
      warn("El navegador no permite la cámara en esta página. Ábrela desde el enlace https del QR.");
      return;
    }
    $("empezar").disabled = true;
    setState("Abriendo cámara", "idle");
    try {
      await startCamera();
    } catch (e) {
      $("empezar").disabled = false;
      setState("Sin cámara", "err");
      warn("No se pudo abrir la cámara: " + (e.message || e.name) + ". Revisa el permiso de cámara para Safari en Ajustes.");
      return;
    }
    ws = await openSocket();
    usePost = !ws;
    $("via").textContent = ws ? "WebSocket" : "HTTPS (POST)";
    timer = setInterval(tick, 1000 / TARGET_FPS);
    keepAwake();
    setState("Transmitiendo", "run");
    $("empezar").hidden = true;
    $("detener").hidden = false;
  }

  function stop() {
    if (timer) clearInterval(timer);
    timer = null;
    if (ws) { var s = ws; ws = null; try { s.close(); } catch (e) {} }
    stopCamera();
    if (wakeLock) { try { wakeLock.release(); } catch (e) {} wakeLock = null; }
    setState("Sin transmitir", "idle");
    $("fps").textContent = "-";
    $("empezar").hidden = false;
    $("empezar").disabled = false;
    $("detener").hidden = true;
  }

  // Safari pausa la camara al bloquear la pantalla o cambiar de app; al volver, se reanuda.
  document.addEventListener("visibilitychange", function () {
    if (document.visibilityState === "visible" && timer) {
      keepAwake();
      if (stream && stream.getVideoTracks().some(function (t) { return t.readyState === "ended"; })) {
        stopCamera();
        startCamera().catch(function () { warn("La cámara se cerró. Pulsa Detener y luego Empezar."); });
      }
    }
  });

  function restartIfRunning() {
    if (!timer) return;
    stopCamera();
    startCamera().catch(function (e) { warn("No se pudo cambiar la cámara: " + e.message); });
  }

  $("empezar").addEventListener("click", start);
  $("detener").addEventListener("click", stop);
  $("lado").addEventListener("change", restartIfRunning);
  $("calidad").addEventListener("change", restartIfRunning);
})();
