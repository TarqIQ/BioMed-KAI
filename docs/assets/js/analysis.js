// BioMed-KAI Analysis JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Add smooth scrolling
    const links = document.querySelectorAll('a[href^="#"]');
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
    
    // Add loading states for interactive content
    console.log('BioMed-KAI Analysis page loaded');
});
