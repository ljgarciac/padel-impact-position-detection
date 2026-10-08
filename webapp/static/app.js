/* Pagina de dos pantallas. El servidor manda instantaneas de la sesion por SSE (~varias por
 * segundo); cada <img> recibe su video anotado por MJPEG. Sin frameworks: solo DOM. */
(function () {
  "use strict";

  var MODELS = ["yolo", "mediapipe"];
  var SPARK_WINDOW_S = 60;
  var $ = function (id) { return document.getElementById(id); };

  var current = null;      // id de la sesion que se esta mostrando
  var events = null;       // EventSource
  var sparks = {};         // modelo -> { fps: [[t, v]], rss: [[t, v]] }
  var videos = [];

  // ---------- formato ----------
  function fmt(v, digits, unit) {
    if (v === null || v === undefined || isNaN(v)) return "-";
    var s = Number(v).toFixed(digits === undefined ? 1 : digits);
    return unit ? s + '<span class="unit">' + unit + "</span>" : s;
  }
  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; });
  }
  function setTag(el, text, kind) {
    el.textContent = text;
    el.className = "tag tag--" + kind;
  }
  var PROC_STATES = {
    starting: ["Arrancando", "idle"], loading: ["Cargando modelo", "run"], waiting: ["Listo", "run"],
    running: ["En vivo", "run"], done: ["Terminado", "ok"], error: ["Error", "err"], died: ["Murió", "err"]
  };
  var SESSION_STATES = {
    queued: ["En cola", "idle"], running: ["En vivo", "run"], stopping: ["Deteniendo", "warn"],
    succeeded: ["Completada", "ok"], failed: ["Falló", "err"], cancelled: ["Cancelada", "idle"]
  };
  function isFinal(state) { return state === "succeeded" || state === "failed" || state === "cancelled"; }

  // ---------- fuentes ----------
  function loadSources() {
    return fetch("/api/sources").then(function (r) { return r.json(); }).then(function (data) {
      videos = data.videos;
      var sel = $("opt-source");
      sel.innerHTML = "";
      videos.forEach(function (v) {
        var o = document.createElement("option");
        o.value = "video:" + v.name;
        o.textContent = "Video: " + v.name;
        sel.appendChild(o);
      });
      [["camera", "Webcam de este PC"], ["phone", "iPhone por QR (misma Wi-Fi)"]].forEach(function (o) {
        var opt = document.createElement("option");
        opt.value = o[0];
        opt.textContent = o[1];
        sel.appendChild(opt);
      });
      if (location.hash === "#iphone") sel.value = "phone";  // enlace directo a la fuente iPhone
      syncSourceFields();
    });
  }
  function syncSourceFields() {
    var val = $("opt-source").value;
    var live = val === "camera" || val === "phone";
    $("cam-field").hidden = val !== "camera";
    $("opt-loops").disabled = live;
    $("opt-pace").disabled = live;
    $("phone-panel").hidden = val !== "phone";
    if (val === "phone") startPhonePolling(); else stopPhonePolling();
  }

  // ---------- iPhone por QR ----------
  var phoneTimer = null, phoneQrUrl = null;
  function startPhonePolling() {
    if (phoneTimer) return;
    pollPhone();
    phoneTimer = setInterval(pollPhone, 1000);
  }
  function stopPhonePolling() {
    if (phoneTimer) clearInterval(phoneTimer);
    phoneTimer = null;
  }
  function pollPhone() {
    fetch("/api/phone").then(function (r) { return r.json(); }).then(function (st) {
      $("phone-error").innerHTML = st.error ? '<div class="notification" role="alert"><strong>' + esc(st.error) + "</strong></div>" : "";
      if (st.url && st.url !== phoneQrUrl) {  // el QR solo se pide de nuevo si cambio el enlace
        phoneQrUrl = st.url;
        $("phone-qr").src = "/api/phone/qr.svg?t=" + Date.now();
      }
      $("phone-url").textContent = st.url || "-";
      if (st.streaming) setTag($("phone-tag"), st.in_session ? "Transmitiendo a la sesión" : "Transmitiendo", "ok");
      else if (st.connected) setTag($("phone-tag"), "Conectado, sin imagen (pulsa Empezar en el iPhone)", "warn");
      else setTag($("phone-tag"), "Esperando al iPhone", "idle");
      $("phone-fps").innerHTML = st.streaming ? fmt(st.fps, 0, "FPS") : "-";
      $("phone-size").textContent = st.size ? st.size[0] + "×" + st.size[1] : "-";
      var prev = $("phone-preview");
      // vista previa solo antes de ejecutar: durante la sesion ya se ve en las dos pantallas
      if (st.streaming && !st.in_session) {
        prev.hidden = false;
        prev.src = "/api/phone/frame?t=" + Date.now();
      } else {
        prev.hidden = true;
      }
    }).catch(function () { /* el servidor se reinicia: el siguiente intento lo resuelve */ });
  }
  function rotatePhoneLink() {
    fetch("/api/phone/rotate", { method: "POST" }).then(function () { phoneQrUrl = null; pollPhone(); });
  }

  // ---------- lanzar / detener ----------
  function start() {
    var models = [];
    if ($("opt-yolo").checked) models.push("yolo");
    if ($("opt-mp").checked) models.push("mediapipe");
    if (!models.length) { showError("Elige al menos un modelo."); return; }
    var srcVal = $("opt-source").value;
    var source = srcVal === "camera" ? { kind: "camera", index: parseInt($("opt-camera").value, 10) || 0 }
      : srcVal === "phone" ? { kind: "phone" }
      : { kind: "video", name: srcVal.slice(6) };
    var body = {
      source: source, models: models, mode: $("opt-mode").value,
      loops: parseInt($("opt-loops").value, 10) || 0, pace_fps: parseFloat($("opt-pace").value) || 0,
      imgsz: parseInt($("opt-imgsz").value, 10), conf: parseFloat($("opt-conf").value),
      mp_variant: $("opt-mpvar").value, preview: $("opt-preview").checked, keypoints: $("opt-kpts").value
    };
    $("btn-start").disabled = true;
    showError("");
    fetch("/api/sessions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
      .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, j: j }; }); })
      .then(function (res) {
        if (!res.ok) {
          $("btn-start").disabled = false;
          var d = res.j.detail;
          showError(typeof d === "string" ? d : "Opciones inválidas: " + JSON.stringify(d));
          return;
        }
        attach(res.j);
      })
      .catch(function (e) { $("btn-start").disabled = false; showError("No se pudo contactar al servidor: " + e); });
  }
  function stop() {
    if (!current) return;
    $("btn-stop").disabled = true;
    fetch("/api/sessions/" + current + "/stop", { method: "POST" });
  }
  function showError(msg, detail) {
    $("p-error").innerHTML = msg ? '<div class="notification" role="alert"><strong>' + esc(msg) + "</strong>" +
      (detail ? "<details><summary>Detalle técnico</summary><pre>" + esc(detail) + "</pre></details>" : "") + "</div>" : "";
  }

  // ---------- seguir una sesion ----------
  function attach(snap) {
    current = snap.id;
    sparks = {};
    MODELS.forEach(function (m) {
      sparks[m] = { fps: [], rss: [] };
      var screen = $("screen-" + m);
      var img = screen.querySelector('[data-f="img"]');
      var on = snap.config.models.indexOf(m) >= 0;
      screen.querySelectorAll("[data-m],[data-r]").forEach(function (dd) { dd.textContent = "-"; });
      screen.querySelector('[data-f="error"]').innerHTML = "";
      if (on && snap.config.preview) {
        img.src = "/api/sessions/" + snap.id + "/stream/" + m + ".mjpg?t=" + Date.now();
        img.hidden = false;
        screen.querySelector('[data-f="empty"]').hidden = true;
      } else {
        img.removeAttribute("src");
        img.hidden = true;
        var empty = screen.querySelector('[data-f="empty"]');
        empty.hidden = false;
        empty.textContent = on ? "Vista en vivo apagada para medir limpio. Las métricas siguen abajo."
          : "Modelo desactivado en esta sesión.";
      }
    });
    $("results").hidden = true;
    if (events) events.close();
    events = new EventSource("/api/sessions/" + snap.id + "/events");
    events.onmessage = function (e) { render(JSON.parse(e.data)); };
    events.addEventListener("end", function () {
      events.close(); events = null;
      fetch("/api/sessions/" + current).then(function (r) { return r.json(); }).then(function (s) { render(s); });
      loadHistory();
    });
    render(snap);
  }

  function render(s) {
    if (s.id !== current) return;
    var st = SESSION_STATES[s.state] || [s.state, "idle"];
    setTag($("s-state"), st[0], st[1]);
    $("s-elapsed").textContent = s.elapsed_s ? s.elapsed_s.toFixed(1) + " s" : "";
    $("s-message").textContent = s.message || "";
    var fin = isFinal(s.state);
    $("btn-start").disabled = !fin;
    $("btn-stop").disabled = fin || s.state === "stopping";

    // progreso: real si hay total; indeterminado si no se conoce (camara, loops = 0)
    var fill = $("p-fill");
    if (fin) { fill.className = "fill"; fill.style.width = s.state === "succeeded" ? "100%" : "0"; }
    else if (s.percent === null || s.percent === undefined) { fill.className = "fill indeterminate"; fill.style.width = ""; }
    else { fill.className = "fill"; fill.style.width = s.percent + "%"; }

    var cap = s.capture || {};
    if (cap.width && cap.height) {
      document.querySelectorAll(".viewport").forEach(function (v) { v.style.aspectRatio = cap.width + " / " + cap.height; });
    }
    var res = s.resources || {};
    var procsRes = res.procs || {};
    MODELS.forEach(function (m) { renderScreen(m, s, procsRes[m], res.t_s); });
    renderSystem(res.system || {}, cap, procsRes.capture, s.procs.capture);

    var capErr = s.procs.capture && s.procs.capture.error;
    if (capErr) showError("La fuente falló: " + capErr.user_message, capErr.technical_detail);
    if (fin) renderResults(s);
  }

  function renderScreen(m, s, r, t) {
    var screen = $("screen-" + m);
    var p = s.procs[m];
    if (!p) { setTag(screen.querySelector('[data-f="state"]'), "Desactivado", "idle"); return; }
    var ps = PROC_STATES[p.state] || [p.state, "idle"];
    setTag(screen.querySelector('[data-f="state"]'), ps[0], ps[1]);
    var dev = screen.querySelector('[data-f="device"]');
    dev.textContent = p.device || "-";
    dev.className = "tag tag--idle";
    var mt = p.metrics || {};
    var set = function (key, html) { var el = screen.querySelector('[data-m="' + key + '"]'); if (el) el.innerHTML = html; };
    set("fps", fmt(mt.fps, 1));
    set("inference_fps", fmt(mt.inference_fps, 1));
    set("inf", fmt(mt.inf_ms_p50, 1) + " / " + fmt(mt.inf_ms_p95, 1, "ms"));
    set("e2e", fmt(mt.e2e_ms_p50, 1) + " / " + fmt(mt.e2e_ms_p95, 1, "ms"));
    set("jitter_ms", fmt(mt.jitter_ms, 1, "ms"));
    set("queue_ms_mean", fmt(mt.queue_ms_mean, 1, "ms"));
    if (mt.frames_processed !== undefined) {
      var total = mt.frames_processed + (mt.frames_dropped || 0);
      set("frames", mt.frames_processed + " / " + (mt.frames_dropped || 0) +
        ' <span class="note">(' + fmt(100 * (mt.frames_dropped || 0) / Math.max(1, total), 0) + " %)</span>");
    }
    set("detections", fmt(mt.detections, 2));
    set("kpt_conf", fmt(mt.kpt_conf, 2));

    var rset = function (key, html) { screen.querySelector('[data-r="' + key + '"]').innerHTML = html; };
    if (r) {
      rset("cpu", fmt(r.cpu_pct, 0, "% de 1 núcleo") + ' <span class="note">(' + fmt(r.cpu_pct_machine, 1) + " % máquina)</span>");
      rset("rss", fmt(r.rss_mb, 0, "MB"));
      rset("threads", String(r.threads));
      if (r.vram_mb !== null && r.vram_mb !== undefined) rset("vram", fmt(r.vram_mb, 0, "MB"));
      else if (mt.vram_reserved_mb !== undefined) rset("vram", fmt(mt.vram_reserved_mb, 0, "MB") + ' <span class="note">(reservada por torch)</span>');
      else rset("vram", m === "mediapipe" ? '<span class="note">no usa GPU (CPU)</span>' : "-");
    }
    if (t !== undefined && !isFinal(s.state)) {
      pushSpark(m, "fps", t, mt.fps);
      if (r) pushSpark(m, "rss", t, r.rss_mb);
    }
    drawSpark(m, "fps", 1, "");
    drawSpark(m, "rss", 0, " MB");

    var err = p.error;
    screen.querySelector('[data-f="error"]').innerHTML = err
      ? '<div class="notification" role="alert"><strong>' + esc(err.user_message || "Error") + "</strong>" +
        (err.technical_detail ? "<details><summary>Detalle técnico</summary><pre>" + esc(err.technical_detail) + "</pre></details>" : "") + "</div>"
      : "";
  }

  function renderSystem(sys, cap, capRes, capProc) {
    $("sys-cpu").innerHTML = fmt(sys.cpu_pct, 0, "%") + (sys.cpus ? ' <span class="note">' + sys.cpus + " núcleos lógicos</span>" : "");
    $("sys-ram").innerHTML = fmt(sys.ram_used_gb, 1) + " / " + fmt(sys.ram_total_gb, 1, "GB");
    $("sys-gpu").innerHTML = sys.gpu_name ? fmt(sys.gpu_pct, 0, "%") + ' <span class="note">' + esc(sys.gpu_name) + "</span>" : '<span class="note">sin GPU NVIDIA</span>';
    $("sys-vram").innerHTML = sys.vram_total_mb ? fmt(sys.vram_used_mb, 0) + " / " + fmt(sys.vram_total_mb, 0, "MB") : "-";
    $("sys-cap").innerHTML = cap.width ? cap.width + "×" + cap.height + " · " + fmt(cap.capture_fps || cap.fps, 1, "FPS") +
      (cap.frame_idx ? ' <span class="note">frame ' + cap.frame_idx + (cap.total ? " de " + cap.total : "") + "</span>" : "") : "-";
    $("sys-capproc").innerHTML = capRes ? fmt(capRes.cpu_pct, 0, "%") + " · " + fmt(capRes.rss_mb, 0, "MB")
      : (capProc ? (PROC_STATES[capProc.state] || [capProc.state])[0] : "-");
  }

  // ---------- sparklines (una serie cada una: sin leyenda, el titulo la nombra) ----------
  function pushSpark(m, key, t, v) {
    if (v === null || v === undefined || isNaN(v)) return;
    var arr = sparks[m][key];
    if (arr.length && arr[arr.length - 1][0] === t) return;
    arr.push([t, v]);
    while (arr.length && t - arr[0][0] > SPARK_WINDOW_S) arr.shift();
  }
  function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function drawSpark(m, key, digits, unit) {
    var fig = $("screen-" + m).querySelector('[data-spark="' + key + '"]');
    var canvas = fig.querySelector("canvas");
    var arr = (sparks[m] || {})[key] || [];
    var dpr = window.devicePixelRatio || 1;
    var w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== w * dpr) { canvas.width = w * dpr; canvas.height = h * dpr; }
    var ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    fig.querySelector("output").innerHTML = arr.length ? fmt(arr[arr.length - 1][1], digits) + unit : "";
    canvas._series = { arr: arr, digits: digits, unit: unit };
    if (arr.length < 2) return;
    var t1 = arr[arr.length - 1][0], t0 = Math.max(arr[0][0], t1 - SPARK_WINDOW_S);
    var vmax = 0;
    arr.forEach(function (p) { vmax = Math.max(vmax, p[1]); });
    vmax = vmax * 1.15 || 1;
    var x = function (t) { return (t - t0) / Math.max(1e-6, t1 - t0) * (w - 2) + 1; };
    var y = function (v) { return h - 3 - (v / vmax) * (h - 8); };
    canvas._scale = { x: x, t0: t0, t1: t1 };
    ctx.strokeStyle = cssVar("--interactive") || "#0f62fe";
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.beginPath();
    arr.forEach(function (p, i) { if (i) ctx.lineTo(x(p[0]), y(p[1])); else ctx.moveTo(x(p[0]), y(p[1])); });
    ctx.stroke();
    if (canvas._hoverX !== undefined) {
      ctx.strokeStyle = cssVar("--border-strong-01");
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(canvas._hoverX, 0); ctx.lineTo(canvas._hoverX, h); ctx.stroke();
    }
  }
  function bindSparkHover() {
    var tip = $("tooltip");
    document.querySelectorAll(".spark canvas").forEach(function (canvas) {
      var fig = canvas.parentNode;
      var m = fig.closest(".screen").dataset.model, key = fig.dataset.spark;
      canvas.addEventListener("mousemove", function (e) {
        var s = canvas._series, sc = canvas._scale;
        if (!s || !sc || s.arr.length < 2) return;
        var rect = canvas.getBoundingClientRect();
        var px = e.clientX - rect.left;
        var best = s.arr[0];
        s.arr.forEach(function (p) { if (Math.abs(sc.x(p[0]) - px) < Math.abs(sc.x(best[0]) - px)) best = p; });
        canvas._hoverX = sc.x(best[0]);
        tip.hidden = false;
        tip.textContent = "t = " + best[0].toFixed(1) + " s · " + Number(best[1]).toFixed(s.digits) + s.unit;
        tip.style.left = (e.clientX + 12) + "px";
        tip.style.top = (e.clientY - 28) + "px";
        drawSpark(m, key, s.digits, s.unit);
      });
      canvas.addEventListener("mouseleave", function () {
        delete canvas._hoverX;
        tip.hidden = true;
        var s = canvas._series;
        if (s) drawSpark(m, key, s.digits, s.unit);
      });
    });
  }

  // ---------- resultados ----------
  var CMP_ROWS = [
    ["pipeline_fps", "FPS pipeline", 1, "max"], ["inference_fps", "FPS inferencia", 1, "max"],
    ["inf_latency_ms_p50", "Inferencia p50 (ms)", 1, "min"], ["inf_latency_ms_p95", "Inferencia p95 (ms)", 1, "min"],
    ["e2e_latency_ms_p50", "Extremo a extremo p50 (ms)", 1, "min"], ["e2e_latency_ms_p95", "Extremo a extremo p95 (ms)", 1, "min"],
    ["jitter_ms", "Jitter (ms)", 1, "min"], ["queue_wait_ms_mean", "Espera en colas (ms)", 1, "min"],
    ["n_frames_steady", "Frames en estado estable", 0, null], ["frames_dropped", "Frames descartados", 0, "min"],
    ["drop_pct", "Descartados (%)", 1, "min"], ["mean_detections", "Personas por frame", 2, null],
    ["mean_kpt_conf", "Confianza de keypoints", 2, null]
  ];
  function renderResults(s) {
    var models = MODELS.filter(function (m) { return s.procs[m] && s.procs[m].summary; });
    if (!models.length) { $("results").hidden = true; return; }
    $("results").hidden = false;
    var names = { yolo: "Nuestro modelo", mediapipe: "MediaPipe Pose" };
    var html = "<thead><tr><th>Métrica</th>" + models.map(function (m) { return '<th class="num">' + names[m] + "</th>"; }).join("") + "</tr></thead><tbody>";
    CMP_ROWS.forEach(function (row) {
      var vals = models.map(function (m) { return s.procs[m].summary[row[0]]; });
      var best = null;
      if (row[3] && models.length > 1) best = row[3] === "max" ? Math.max.apply(null, vals) : Math.min.apply(null, vals);
      html += "<tr><td>" + row[1] + "</td>" + vals.map(function (v) {
        return '<td class="num' + (best !== null && v === best ? " better" : "") + '">' + fmt(v, row[2]) + "</td>";
      }).join("") + "</tr>";
    });
    html += "<tr><td>Dispositivo</td>" + models.map(function (m) { return '<td class="num">' + esc(s.procs[m].summary.device || "-") + "</td>"; }).join("") + "</tr>";
    $("cmp").innerHTML = html + "</tbody>";

    $("outputs").innerHTML = models.map(function (m) {
      var p = s.procs[m], base = "/api/sessions/" + s.id + "/files/" + m;
      var vid = p.files && p.files.video
        ? '<video controls preload="metadata" src="' + base + '/video"></video>'
        : '<p class="meta">Sin video: ' + esc(p.video_error || "no se generó") + "</p>";
      return "<div><h3>" + names[m] + "</h3>" + vid + '<div class="links"><a href="' + base + '/csv">Descargar CSV por frame</a></div></div>';
    }).join("") + '<div class="links"><a href="/api/sessions/' + s.id + '/files/recursos">Descargar recursos.csv (CPU, RAM, GPU en el tiempo)</a></div>';
  }

  // ---------- historial ----------
  function loadHistory() {
    fetch("/api/sessions").then(function (r) { return r.json(); }).then(function (list) {
      var html = '<thead><tr><th>Hora</th><th>Fuente</th><th>Modo</th><th>Estado</th>' +
        '<th class="num">FPS propio</th><th class="num">E2E p50 propio (ms)</th>' +
        '<th class="num">FPS MediaPipe</th><th class="num">E2E p50 MediaPipe (ms)</th></tr></thead><tbody>';
      if (!list.length) html += '<tr><td colspan="8" class="meta">Todavía no hay sesiones. Lanza una con Ejecutar.</td></tr>';
      list.forEach(function (s) {
        var k = s.config.source.kind;
        var src = k === "camera" ? "Webcam " + s.config.source.index : k === "phone" ? "iPhone" : s.config.source.name;
        var sm = function (m, k, d) { var p = s.procs[m]; return p && p.summary ? fmt(p.summary[k], d) : "-"; };
        html += "<tr><td class=\"num\">" + new Date(s.created_at * 1000).toLocaleTimeString() + "</td><td>" + esc(src) +
          "</td><td>" + s.config.mode + "</td><td>" + (SESSION_STATES[s.state] || [s.state])[0] + "</td>" +
          '<td class="num">' + sm("yolo", "pipeline_fps", 1) + '</td><td class="num">' + sm("yolo", "e2e_latency_ms_p50", 1) +
          '</td><td class="num">' + sm("mediapipe", "pipeline_fps", 1) + '</td><td class="num">' + sm("mediapipe", "e2e_latency_ms_p50", 1) + "</td></tr>";
      });
      $("history").innerHTML = html + "</tbody>";
      // si se recarga la pagina con una sesion en curso, se retoma
      if (!current) {
        var live = list.filter(function (s) { return !isFinal(s.state); })[0];
        if (live) attach(live);
      }
    });
  }

  $("opt-source").addEventListener("change", syncSourceFields);
  $("btn-start").addEventListener("click", start);
  $("btn-stop").addEventListener("click", stop);
  $("phone-rotate").addEventListener("click", rotatePhoneLink);
  bindSparkHover();
  loadSources().then(loadHistory);
})();
