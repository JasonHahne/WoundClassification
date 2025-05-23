document.addEventListener('DOMContentLoaded', () => {
    // Disclaimer Logic
    const disclaimerModal = document.getElementById('disclaimerModal');
    const disclaimerCheckbox = document.getElementById('disclaimerCheckbox');
    const disclaimerButton = document.getElementById('disclaimerButton');

    if (!localStorage.getItem('disclaimerAccepted')) {
        disclaimerModal.style.display = 'flex';
        document.body.style.overflow = 'hidden';
    }

    disclaimerCheckbox.addEventListener('change', () => {
        disclaimerButton.disabled = !disclaimerCheckbox.checked;
    });

    disclaimerButton.addEventListener('click', () => {
        localStorage.setItem('disclaimerAccepted', 'true');
        disclaimerModal.style.display = 'none';
        document.body.style.overflow = 'auto';
    });

    disclaimerModal.addEventListener('click', (e) => {
        if (e.target === disclaimerModal) {
            disclaimerModal.style.display = 'none';
            document.body.style.overflow = 'auto';
        }
    });

    // Original Application Logic
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const previewImage = document.getElementById('previewImage');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const analyzeButton = document.getElementById('analyzeButton');
    const resultsDiv = document.getElementById('results');

    // Create clear button
    const clearBtn = document.createElement('button');
    clearBtn.innerHTML = '&times;';
    clearBtn.className = 'clear-btn';
    clearBtn.title = 'Remove image';
    clearBtn.onclick = () => {
        fileInput.value = '';
        dropZone.classList.remove('has-image');
        previewImage.style.display = 'none';
        resultsDiv.innerHTML = '';
    };
    dropZone.appendChild(clearBtn);

    // Drag & drop handlers
    const preventDefaults = (e) => {
        e.preventDefault();
        e.stopPropagation();
    };

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(event => {
        dropZone.addEventListener(event, preventDefaults, false);
    });

    dropZone.addEventListener('dragover', () => {
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files[0]) handleFile(files[0]);
        dropZone.classList.remove('dragover');
    });

    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files[0]) {
            handleFile(e.target.files[0]);
        } else {
            dropZone.classList.remove('has-image');
            previewImage.style.display = 'none';
            resultsDiv.innerHTML = '';
        }
    });

    // Form submission
    document.getElementById('uploadForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!fileInput.files[0]) return;

        loadingOverlay.style.display = 'flex';
        analyzeButton.disabled = true;

        try {
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || 'Server error');
            }

            displayResults(result.predictions);

        } catch (error) {
            showError(error.message);
        } finally {
            loadingOverlay.style.display = 'none';
            analyzeButton.disabled = false;
        }
    });

    function handleFile(file) {
        if (file.size > 5 * 1024 * 1024) {
            showError('File size exceeds 5MB limit');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            dropZone.classList.add('has-image');
            previewImage.src = e.target.result;
            previewImage.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }

    function displayResults(predictions) {
        const resultsDiv = document.getElementById('results');
        resultsDiv.innerHTML = predictions.map(pred => `
            <div class="result-item">
                <span class="class-name">${pred.class}</span>
                <div class="confidence-bar">
                    <div class="confidence-fill"
                         style="width: ${parseFloat(pred.confidence)}%">
                    </div>
                </div>
                <span class="confidence-value">${pred.confidence}</span>
            </div>
        `).join('');
    }

    function showError(message) {
        const resultsDiv = document.getElementById('results');
        resultsDiv.innerHTML = `
            <div class="error-message">
                ⚠️ ${message}
            </div>
        `;
    }
});