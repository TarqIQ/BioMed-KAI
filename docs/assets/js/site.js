document.addEventListener('DOMContentLoaded', () => {
    const mobileToggle = document.querySelector('[data-nav-toggle]');
    const mobileMenu = document.querySelector('[data-mobile-nav]');

    if (mobileToggle && mobileMenu) {
        mobileToggle.addEventListener('click', () => {
            mobileMenu.classList.toggle('hidden');
        });
    }

    // Smooth scrolling for in-page anchors
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', event => {
            const targetId = anchor.getAttribute('href');
            if (targetId.length > 1) {
                const targetEl = document.querySelector(targetId);
                if (targetEl) {
                    event.preventDefault();
                    targetEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            }
        });
    });

    // Copy-to-clipboard buttons
    document.querySelectorAll('[data-copy-target]').forEach(button => {
        button.addEventListener('click', async () => {
            const targetSelector = button.getAttribute('data-copy-target');
            const target = document.querySelector(targetSelector);
            if (!target) return;

            try {
                await navigator.clipboard.writeText(target.innerText.trim());
                const original = button.innerHTML;
                button.innerHTML = '<i class="fas fa-check mr-2"></i>Copied!';
                setTimeout(() => (button.innerHTML = original), 1500);
            } catch (err) {
                console.error('Copy failed', err);
                alert('Copy failed. Please copy manually.');
            }
        });
    });
});

