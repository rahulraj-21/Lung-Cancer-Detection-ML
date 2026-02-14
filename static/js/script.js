// File upload handling
document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.querySelector('.upload-area');
    const fileName = document.getElementById('fileName');
    
    if (uploadArea && fileInput) {
        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.background = 'rgba(255, 255, 255, 0.15)';
            uploadArea.style.borderColor = '#4caf50';
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.style.background = 'rgba(255, 255, 255, 0.05)';
            uploadArea.style.borderColor = '#ff9800';
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.background = 'rgba(255, 255, 255, 0.05)';
            uploadArea.style.borderColor = '#ff9800';
            
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                updateFileName();
            }
        });
        
        fileInput.addEventListener('change', updateFileName);
    }
    
    function updateFileName() {
        if (fileInput.files.length > 0) {
            fileName.textContent = fileInput.files[0].name;
            fileName.style.color = '#4caf50';
        } else {
            fileName.textContent = 'No file chosen';
            fileName.style.color = '#ff9800';
        }
    }
    
    // Confidence meter animation
    const confidenceMeter = document.getElementById('confidenceMeter');
    if (confidenceMeter) {
        const confidence = parseFloat(confidenceMeter.getAttribute('data-confidence'));
        setTimeout(() => {
            confidenceMeter.style.width = `${confidence * 100}%`;
        }, 500);
    }
    
    // Auto-dismiss alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000);
    });
});

// Chart initialization for dashboard
function initDashboardCharts() {
    const ctx = document.getElementById('predictionChart');
    if (ctx) {
        const chart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Normal', 'Benign', 'Malignant'],
                datasets: [{
                    data: [65, 25, 10], // Example data
                    backgroundColor: [
                        '#4caf50',
                        '#ff9800',
                        '#f44336'
                    ],
                    borderWidth: 2,
                    borderColor: '#1a237e'
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#ffffff',
                            font: {
                                size: 14
                            }
                        }
                    }
                }
            }
        });
    }
}

// Initialize charts when page loads
document.addEventListener('DOMContentLoaded', initDashboardCharts);