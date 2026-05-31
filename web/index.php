<?php
// web/index.php
$maxSizeMB   = 50;
$allowedExts = ['wav', 'mp3'];
?>
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Müzik Türü Sınıflandırma</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css">
    <link href="assets/css/style.css?v=2" rel="stylesheet">
</head>
<body>
<div class="container py-5">
    <div class="text-center mb-5">
        <h1 class="display-4 fw-bold text-white mb-3"><i class="bi bi-music-note-beamed text-info"></i> Müzik Analizi</h1>
        <p class="lead text-light opacity-75">Ses dosyanızı yükleyin, sistem müzikal karakteri çıkartsın ve analiz etsin.</p>
    </div>

    <div class="row justify-content-center">
        <div class="col-lg-8">
            <div class="glass-card mb-4">
                <form id="uploadForm" action="upload.php" method="POST" enctype="multipart/form-data">
                    <div class="upload-zone" id="dropZone">
                        <i class="bi bi-cloud-arrow-up upload-icon"></i>
                        <h4>Şarkınızı Buraya Sürükleyin</h4>
                        <p class="text-muted mb-3">veya seçmek için tıklayın (.wav, .mp3)</p>
                        <input type="file" id="audioFile" name="audioFile" accept=".wav,.mp3" required>
                        <div id="fileInfo" class="mt-2 text-info fw-bold"></div>
                    </div>
                    
                    <div id="playerContainer" class="mt-4 d-none text-center">
                        <p class="mb-1 text-light">Seçilen Şarkı:</p>
                        <audio id="previewAudio" controls></audio>
                    </div>

                    <div class="text-center mt-4">
                        <button type="submit" class="btn btn-glass btn-lg w-75" id="submitBtn" disabled>
                            <i class="bi bi-cpu"></i> Analizi Başlat
                        </button>
                    </div>
                </form>

                <div id="progressSection" class="mt-4 d-none text-center">
                    <p class="text-info mb-2"><i class="bi bi-hourglass-split"></i> Ses frekansları analiz ediliyor...</p>
                    <div class="progress" style="height: 10px;">
                        <div class="progress-bar progress-bar-striped progress-bar-animated w-100" role="progressbar"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Sonuç Bölümü -->
    <div id="resultSection" class="row mt-2 d-none">
        
        <!-- Sol Kolon: Tür & Spektrogram -->
        <div class="col-lg-7 mb-4">
            <div class="glass-card h-100">
                <h3 class="text-center mb-4 text-white border-bottom border-secondary pb-3">
                    Analiz Sonucu: <span id="resGenre" class="badge badge-genre text-uppercase ms-2"></span>
                </h3>
                
                <div class="mb-4">
                    <div class="d-flex justify-content-between text-light mb-1">
                        <span>Model Güven Skoru</span>
                        <span id="resConfText" class="fw-bold text-info"></span>
                    </div>
                    <div class="progress" style="height: 12px;">
                        <div id="resConfBar" class="progress-bar bg-info" role="progressbar"></div>
                    </div>
                </div>

                <!-- Karar Özeti (XAI) -->
                <div class="mt-4 p-3 rounded" style="background: rgba(0, 242, 254, 0.05); border-left: 4px solid #00f2fe;">
                    <h6 class="text-info mb-2"><i class="bi bi-bar-chart-steps"></i> Karar Özeti</h6>
                    <p id="resExplanation" class="text-light mb-0" style="font-size: 0.95rem; line-height: 1.5;"></p>
                </div>
            </div>
        </div>

        <!-- Sağ Kolon: Radar Chart & Öneriler -->
        <div class="col-lg-5 mb-4">
            <div class="glass-card mb-4">
                <h5 class="text-center text-light mb-3"><i class="bi bi-pentagon"></i> Akustik Karakteristik</h5>
                <div style="position: relative; height: 300px; width: 100%;">
                    <canvas id="radarChart"></canvas>
                </div>
            </div>

            <div class="glass-card">
                <h5 class="text-center text-light mb-3"><i class="bi bi-headphones"></i> Benzer Şarkı Önerileri</h5>
                <div id="resRecommendations"></div>
            </div>
        </div>

    </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="assets/js/app.js?v=4"></script>
</body>
</html>
