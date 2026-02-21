(function () {
  'use strict';

  // ----- Particle background (canvas) -----
  function initParticles() {
    var canvas = document.getElementById('particlesCanvas');
    if (!canvas) return;

    var ctx = canvas.getContext('2d');
    var particles = [];
    var particleCount = 60;
    var connectionDistance = 120;

    function resize() {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      if (particles.length === 0) {
        for (var i = 0; i < particleCount; i++) {
          particles.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            vx: (Math.random() - 0.5) * 0.4,
            vy: (Math.random() - 0.5) * 0.4,
            r: 1.2 + Math.random() * 1.5,
            opacity: 0.15 + Math.random() * 0.2
          });
        }
      }
    }

    function animate() {
      if (!ctx || !canvas.width) return requestAnimationFrame(animate);
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      for (var i = 0; i < particles.length; i++) {
        var p = particles[i];
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
        p.x = Math.max(0, Math.min(canvas.width, p.x));
        p.y = Math.max(0, Math.min(canvas.height, p.y));

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(124, 58, 237, ' + p.opacity + ')';
        ctx.fill();
      }

      ctx.strokeStyle = 'rgba(124, 58, 237, 0.06)';
      ctx.lineWidth = 0.8;
      for (var i = 0; i < particles.length; i++) {
        for (var j = i + 1; j < particles.length; j++) {
          var dx = particles[i].x - particles[j].x;
          var dy = particles[i].y - particles[j].y;
          var d = Math.sqrt(dx * dx + dy * dy);
          if (d < connectionDistance) {
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.stroke();
          }
        }
      }

      requestAnimationFrame(animate);
    }

    resize();
    window.addEventListener('resize', resize);
    requestAnimationFrame(animate);
  }

  // ----- Parallax (hero section) -----
  function initParallax() {
    var heroCopy = document.getElementById('heroCopy');
    var heroPanel = document.getElementById('heroPanel');
    if (!heroCopy && !heroPanel) return;

    document.addEventListener('mousemove', function (e) {
      var cx = window.innerWidth / 2;
      var cy = window.innerHeight / 2;
      var dx = (e.clientX - cx) / cx;
      var dy = (e.clientY - cy) / cy;
      var move = 8;
      if (heroCopy) {
        heroCopy.style.transform = 'translate(' + dx * move + 'px, ' + dy * move + 'px)';
      }
      if (heroPanel) {
        heroPanel.style.transform = 'translate(' + -dx * (move * 0.6) + 'px, ' + -dy * (move * 0.6) + 'px)';
      }
    });
  }

  // ----- Upload: preview, drag-drop, loading, scanner -----
  function initUpload() {
    var fileInput = document.getElementById('fileInput');
    var uploadArea = document.getElementById('uploadArea');
    var fileInfo = document.getElementById('fileInfo');
    var uploadForm = document.getElementById('uploadForm');
    var submitBtn = document.getElementById('submitBtn');
    var loader = document.getElementById('loader');
    var browseButton = document.getElementById('browseButton');
    var previewShell = document.getElementById('previewShell');
    var imagePreview = document.getElementById('imagePreview');
    var scannerOverlay = document.getElementById('scannerOverlay');

    function setFileInfo(file) {
      if (!fileInfo) return;
      fileInfo.textContent = 'Selected: ' + file.name + ' · ' + (file.size / 1024 / 1024).toFixed(2) + ' MB';
    }

    function showPreview(file) {
      if (!imagePreview || !previewShell) return;
      if (!file || !file.type.startsWith('image/')) return;
      var reader = new FileReader();
      reader.onload = function (e) {
        imagePreview.src = e.target.result;
        previewShell.setAttribute('aria-hidden', 'false');
        if (scannerOverlay) scannerOverlay.setAttribute('aria-hidden', 'false');
      };
      reader.readAsDataURL(file);
    }

    if (browseButton && fileInput) {
      browseButton.addEventListener('click', function (e) {
        e.preventDefault();
        fileInput.click();
      });
    }

    if (fileInput) {
      fileInput.addEventListener('change', function (e) {
        var file = e.target.files[0];
        if (file) {
          setFileInfo(file);
          showPreview(file);
          if (uploadArea) uploadArea.classList.add('has-file');
        }
      });
    }

    if (uploadArea && fileInput) {
      ['dragenter', 'dragover'].forEach(function (eventName) {
        uploadArea.addEventListener(eventName, function (e) {
          e.preventDefault();
          e.stopPropagation();
          uploadArea.classList.add('dragover');
        });
      });
      ['dragleave', 'drop'].forEach(function (eventName) {
        uploadArea.addEventListener(eventName, function (e) {
          e.preventDefault();
          e.stopPropagation();
          uploadArea.classList.remove('dragover');
        });
      });
      uploadArea.addEventListener('drop', function (e) {
        var files = e.dataTransfer.files;
        if (files && files.length > 0) {
          var file = files[0];
          fileInput.files = files;
          setFileInfo(file);
          showPreview(file);
          uploadArea.classList.add('has-file');
        }
      });
      uploadArea.addEventListener('click', function () {
        if (fileInput) fileInput.click();
      });
    }

    if (uploadForm && submitBtn && fileInput) {
      uploadForm.addEventListener('submit', function (e) {
        if (!fileInput.files.length) {
          e.preventDefault();
          alert('Please select an image file');
          return;
        }
        submitBtn.disabled = true;
        submitBtn.classList.add('is-loading');
        var textSpan = submitBtn.querySelector('.btn-text');
        if (textSpan) textSpan.textContent = 'Analyzing with AI...';
      });
    }
  }

  // ----- Result page: animate confidence bar -----
  function initResultPage() {
    var meterFill = document.querySelector('.meter-fill');
    if (meterFill && meterFill.dataset && meterFill.dataset.confidence) {
      var confidence = parseFloat(meterFill.dataset.confidence);
      requestAnimationFrame(function () {
        meterFill.style.width = Math.max(1, Math.min(confidence, 100)) + '%';
      });
    }
  }

  // ----- Init on DOM ready -----
  function init() {
    initParticles();
    initParallax();
    initUpload();
    initResultPage();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
