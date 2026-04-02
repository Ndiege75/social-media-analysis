// Platform Usage Chart
const platformCtx = document.getElementById('platformChart').getContext('2d');
new Chart(platformCtx, {
    type: 'bar',
    data: {
        labels: ['TikTok', 'Instagram', 'Facebook', 'Twitter', 'LinkedIn'],
        datasets: [{
            label: 'Average Hours per Day',
            data: [2.8, 2.3, 1.5, 0.8, 0.5],
            backgroundColor: [
                'rgba(54, 162, 235, 0.8)',
                'rgba(153, 102, 255, 0.8)',
                'rgba(75, 192, 192, 0.8)',
                'rgba(255, 159, 64, 0.8)',
                'rgba(255, 99, 132, 0.8)'
            ],
            borderColor: [
                'rgba(54, 162, 235, 1)',
                'rgba(153, 102, 255, 1)',
                'rgba(75, 192, 192, 1)',
                'rgba(255, 159, 64, 1)',
                'rgba(255, 99, 132, 1)'
            ],
            borderWidth: 1
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
            legend: { position: 'top' },
            title: { display: true, text: 'Platform Usage Distribution' }
        },
        scales: {
            y: { beginAtZero: true, title: { display: true, text: 'Hours per Day' } }
        }
    }
});

// Age Group Distribution Chart
const ageCtx = document.getElementById('ageChart').getContext('2d');
new Chart(ageCtx, {
    type: 'pie',
    data: {
        labels: ['18-24', '25-34', '35-44', '13-17', '45-54', '55+'],
        datasets: [{
            data: [25, 30, 20, 5, 12, 8],
            backgroundColor: [
                '#3498db', '#2ecc71', '#f39c12', 
                '#e74c3c', '#9b59b6', '#1abc9c'
            ]
        }]
    },
    options: {
        responsive: true,
        plugins: {
            legend: { position: 'right' },
            title: { display: true, text: 'Age Group Distribution' }
        }
    }
});

// Gender Distribution Chart
const genderCtx = document.getElementById('genderChart').getContext('2d');
new Chart(genderCtx, {
    type: 'doughnut',
    data: {
        labels: ['Female', 'Male', 'Non-binary'],
        datasets: [{
            data: [48, 48, 4],
            backgroundColor: ['#e84393', '#3498db', '#f39c12']
        }]
    },
    options: {
        responsive: true,
        plugins: {
            legend: { position: 'bottom' },
            title: { display: true, text: 'Gender Distribution' }
        }
    }
});

// Add animation to cards on scroll
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

document.querySelectorAll('.segment-card, .insight-card').forEach(card => {
    card.style.opacity = '0';
    card.style.transform = 'translateY(20px)';
    card.style.transition = 'all 0.6s ease-out';
    observer.observe(card);
});