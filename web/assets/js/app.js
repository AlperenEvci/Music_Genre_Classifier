// web/assets/js/app.js
'use strict';

document.getElementById('uploadForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const submitBtn       = document.getElementById('submitBtn');
    const progressSection = document.getElementById('progressSection');
    const resultSection   = document.getElementById('resultSection');

    submitBtn.disabled = true;
    progressSection.classList.remove('d-none');
    resultSection.innerHTML = '';

    const formData = new FormData(this);

    try {
        const response = await fetch('upload.php', { method: 'POST', body: formData });
        const data = await response.json();
        renderResult(data);
    } catch (err) {
        resultSection.innerHTML =
            `<div class="col-12"><div class="alert alert-danger">İstek başarısız: ${escapeHtml(err.message)}</div></div>`;
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
    const section = document.getElementById('resultSection');

    if (data.error) {
        section.innerHTML =
            `<div class="col-12"><div class="alert alert-danger">${escapeHtml(data.error)}</div></div>`;
        return;
    }

    const confidence = (data.confidence * 100).toFixed(1);
    const recsHtml = (data.recommendations || []).map(r =>
        `<li class="list-group-item bg-dark text-light border-secondary">
            <strong>${escapeHtml(r.genre)}</strong> — ${escapeHtml(r.file)}
            <span class="badge bg-info float-end">${(r.score * 100).toFixed(1)}% benzer</span>
         </li>`
    ).join('');

    section.innerHTML = `
        <div class="col-md-8">
            <div class="card bg-secondary text-light p-4">
                <h3 class="text-center mb-3">
                    Tür: <span class="badge bg-success fs-4">${escapeHtml(data.genre.toUpperCase())}</span>
                </h3>
                <div class="mb-3">
                    <label class="form-label">Güven Skoru</label>
                    <div class="progress" style="height:24px;">
                        <div class="progress-bar bg-success" style="width:${escapeHtml(confidence)}%">${escapeHtml(confidence)}%</div>
                    </div>
                </div>
                ${recsHtml ? `
                <h5 class="mt-3">Benzer Şarkılar</h5>
                <ul class="list-group">${recsHtml}</ul>` : ''}
            </div>
        </div>`;
}
