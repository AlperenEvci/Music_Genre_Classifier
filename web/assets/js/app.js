// web/assets/js/app.js
'use strict';

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('audioFile');
const fileInfo = document.getElementById('fileInfo');
const submitBtn = document.getElementById('submitBtn');
const previewAudio = document.getElementById('previewAudio');
const playerContainer = document.getElementById('playerContainer');
let radarChartInstance = null;

// Sürükle Bırak Olayları
dropZone.addEventListener('click', () => fileInput.click());

['dragover', 'dragenter'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
});

['dragleave', 'dragend', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
    });
});

dropZone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        handleFileSelect();
    }
});

fileInput.addEventListener('change', handleFileSelect);

function handleFileSelect() {
    if (fileInput.files.length > 0) {
        const file = fileInput.files[0];
        fileInfo.innerHTML = `<i class="bi bi-music-note"></i> ${escapeHtml(file.name)} (${(file.size / (1024*1024)).toFixed(2)} MB)`;
        submitBtn.disabled = false;
        
        // Önizleme için ses yükle
        const objectUrl = URL.createObjectURL(file);
        previewAudio.src = objectUrl;
        playerContainer.classList.remove('d-none');
    }
}

document.getElementById('uploadForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const progressSection = document.getElementById('progressSection');
    const resultSection   = document.getElementById('resultSection');

    submitBtn.disabled = true;
    progressSection.classList.remove('d-none');
    resultSection.classList.add('d-none');

    const formData = new FormData(this);

    try {
        const response = await fetch('upload.php', { method: 'POST', body: formData });
        const data = await response.json();
        renderResult(data);
    } catch (err) {
        alert("İstek başarısız: " + err.message);
    } finally {
        submitBtn.disabled = false;
        progressSection.classList.add('d-none');
    }
});

function escapeHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

function renderResult(data) {
    if (data.error) {
        alert(data.error);
        return;
    }

    const section = document.getElementById('resultSection');
    section.classList.remove('d-none');

    // Tür ve Güven
    document.getElementById('resGenre').innerText = data.genre;
    const conf = (data.confidence * 100).toFixed(1);
    document.getElementById('resConfText').innerText = conf + '%';
    document.getElementById('resConfBar').style.width = conf + '%';

    // XAI (Yapay Zeka Karar Özeti)
    const explanationEl = document.getElementById('resExplanation');
    if (data.explanation) {
        explanationEl.innerHTML = data.explanation;
        explanationEl.parentElement.style.display = 'block';
    } else {
        explanationEl.parentElement.style.display = 'none';
    }

    // Öneriler (Ses oynatıcı eklendi!)
    const recsContainer = document.getElementById('resRecommendations');
    if (data.recommendations && data.recommendations.length > 0) {
        recsContainer.innerHTML = data.recommendations.map(r => `
            <div class="rec-item">
                <div class="d-flex justify-content-between text-light">
                    <span><span class="badge bg-secondary">${escapeHtml(r.genre)}</span> ${escapeHtml(r.file)}</span>
                    <span class="text-info">${(r.score * 100).toFixed(1)}% Benzer</span>
                </div>
                <!-- Flask API üzerinden orijinal dosyayı çal -->
                <audio controls src="http://localhost:5000/audio/${escapeHtml(r.genre)}/${escapeHtml(r.file)}"></audio>
            </div>
        `).join('');
    } else {
        recsContainer.innerHTML = '<p class="text-muted">Öneri bulunamadı.</p>';
    }

    // Radar Grafiği (Chart.js)
    renderChart(data.features);
}

function renderChart(features) {
    const ctx = document.getElementById('radarChart').getContext('2d');
    
    if (radarChartInstance) {
        radarChartInstance.destroy();
    }

    if (!features) return;

    // Verilerin Radar için normalize edilmiş / orantılanmış görünümü
    // Gerçek değerlerin farklı ölçekleri olduğu için Radar'da düzgün durması adına 100 üzerinden scale ediyoruz.
    // Bu sadece görselleştirme amaçlıdır.
    const normData = [
        Math.min((features.tempo || 0) / 200 * 100, 100),            // Tempo (max ~200)
        Math.min((features.sc_mean || 0) / 4000 * 100, 100),         // Parlaklık / Centroid (max ~4000)
        Math.min((features.rms_mean || 0) / 0.5 * 100, 100),         // Enerji / RMS (max ~0.5)
        Math.min((features.zcr_mean || 0) / 0.2 * 100, 100),         // Pürüzlülük / ZCR (max ~0.2)
        Math.min((features.chroma_mean || 0) / 0.6 * 100, 100)       // Harmonik / Chroma (max ~0.6)
    ];

    radarChartInstance = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Tempo/Hız', 'Parlaklık (Tiz)', 'Ses Şiddeti/Enerji', 'Pürüzlülük (Metalik)', 'Harmonik Yapı'],
            datasets: [{
                label: 'Akustik Profil',
                data: normData,
                backgroundColor: 'rgba(0, 242, 254, 0.2)',
                borderColor: '#00f2fe',
                pointBackgroundColor: '#f093fb',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: '#f093fb',
                borderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    pointLabels: { color: '#e0e0e0', font: { size: 11, family: 'Inter' } },
                    ticks: { display: false, min: 0, max: 100 }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}
