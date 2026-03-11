/* ============================================================
   Blood Bank Management System — Main JavaScript
   ============================================================ */

'use strict';

// ── State ──────────────────────────────────────────────────
let currentRole = 'admin';
let charts = {};

// ── Initialise ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  setupRoleTabs();
  setupLoginForm();
});

// ── Role Tabs ──────────────────────────────────────────────
function setupRoleTabs() {
  const tabs = document.querySelectorAll('.role-tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      currentRole = tab.dataset.role;
    });
  });
}

// ── Login ──────────────────────────────────────────────────
function setupLoginForm() {
  const form = document.getElementById('login-form');
  if (!form) return;
  form.addEventListener('submit', e => {
    e.preventDefault();
    handleLogin();
  });
}

function handleLogin() {
  const btn = document.getElementById('login-btn');
  btn.textContent = 'Signing in…';
  btn.disabled = true;

  setTimeout(() => {
    btn.textContent = 'Sign In';
    btn.disabled = false;
    showPage('admin-dashboard');
    updateSidebarRole(currentRole);
    initCharts();
  }, 900);
}

// ── Page Navigation ────────────────────────────────────────
function showPage(pageId) {
  // Switch between login and main layout
  document.getElementById('page-login').classList.remove('active');
  document.getElementById('main-layout').classList.remove('hidden');
  document.getElementById('main-layout').style.display = 'flex';

  navigateToContent(pageId);
}

function navigate(linkEl) {
  if (!linkEl) return;
  const pageId = linkEl.dataset.page;
  if (!pageId) return;

  // Update active nav link
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  linkEl.classList.add('active');

  navigateToContent(pageId);
}

function navigateToContent(pageId) {
  // Hide all content pages
  document.querySelectorAll('.content-page').forEach(p => p.classList.remove('active'));

  const target = document.getElementById(pageId);
  if (target) {
    target.classList.add('active');
    updatePageTitle(pageId);

    // Lazy-init charts on dashboard
    if (pageId === 'admin-dashboard') {
      initCharts();
    }

    // Scroll to top
    const wrapper = document.querySelector('.content-wrapper');
    if (wrapper) wrapper.scrollTop = 0;
  }

  // Sync sidebar active state
  document.querySelectorAll('.nav-item').forEach(n => {
    n.classList.toggle('active', n.dataset.page === pageId);
  });
}

const pageTitles = {
  'admin-dashboard':    'Dashboard',
  'blood-inventory':    'Blood Inventory',
  'donor-registration': 'Donor Registration',
  'hospital-request':   'Blood Request',
  'emergency-alerts':   'Emergency Alerts',
};

function updatePageTitle(pageId) {
  const el = document.getElementById('page-title');
  if (el) el.textContent = pageTitles[pageId] || 'Dashboard';
}

// ── Sidebar Role ───────────────────────────────────────────
function updateSidebarRole(role) {
  const roleEl = document.getElementById('sidebar-role');
  if (!roleEl) return;
  const roleMap = {
    admin:    'Administrator',
    hospital: 'Hospital Staff',
    donor:    'Donor',
  };
  roleEl.textContent = roleMap[role] || 'Administrator';
}

// ── Sidebar Toggle (mobile) ────────────────────────────────
function toggleSidebar() {
  document.getElementById('sidebar').classList.toggle('open');
}

// ── Charts ─────────────────────────────────────────────────
function initCharts() {
  if (charts.bloodStock) return; // Already initialised

  const bloodGroups = ['A+', 'A−', 'B+', 'B−', 'O+', 'O−', 'AB+', 'AB−'];
  const units       = [620, 180, 540, 95, 890, 210, 430, 249];

  // Blood stock bar chart
  const ctx1 = document.getElementById('bloodStockChart');
  if (ctx1) {
    charts.bloodStock = new Chart(ctx1, {
      type: 'bar',
      data: {
        labels: bloodGroups,
        datasets: [{
          label: 'Units Available',
          data: units,
          backgroundColor: units.map(u =>
            u < 150 ? 'rgba(220,38,38,0.85)' :
            u < 300 ? 'rgba(217,119,6,0.85)' :
                      'rgba(22,163,74,0.85)'
          ),
          borderRadius: 6,
          borderSkipped: false,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: ctx => ` ${ctx.parsed.y} units`,
            },
          },
        },
        scales: {
          x: { grid: { display: false }, ticks: { font: { size: 12, weight: '600' } } },
          y: {
            grid: { color: '#F3F4F6' },
            ticks: { font: { size: 11 } },
            beginAtZero: true,
          },
        },
      },
    });
  }

  // Monthly donations line chart
  const ctx2 = document.getElementById('donationTrendChart');
  if (ctx2) {
    charts.donationTrend = new Chart(ctx2, {
      type: 'line',
      data: {
        labels: ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar'],
        datasets: [
          {
            label: 'Donations Collected',
            data: [380, 420, 510, 460, 530, 490],
            borderColor: '#DC2626',
            backgroundColor: 'rgba(220,38,38,0.08)',
            fill: true,
            tension: 0.4,
            pointBackgroundColor: '#DC2626',
            pointRadius: 4,
            pointHoverRadius: 6,
          },
          {
            label: 'Blood Requests',
            data: [310, 380, 450, 390, 460, 420],
            borderColor: '#3B82F6',
            backgroundColor: 'rgba(59,130,246,0.06)',
            fill: true,
            tension: 0.4,
            pointBackgroundColor: '#3B82F6',
            pointRadius: 4,
            pointHoverRadius: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
            labels: { font: { size: 12 }, usePointStyle: true, pointStyleWidth: 8 },
          },
        },
        scales: {
          x: { grid: { display: false }, ticks: { font: { size: 11 } } },
          y: {
            grid: { color: '#F3F4F6' },
            ticks: { font: { size: 11 } },
            beginAtZero: true,
          },
        },
      },
    });
  }
}

// ── Forms ──────────────────────────────────────────────────
function submitDonorForm(e) {
  e.preventDefault();
  showToast('✅ Donor registered successfully!', 3000);
  e.target.reset();
}

function resetDonorForm() {
  document.getElementById('donor-form').reset();
}

function submitRequestForm(e) {
  e.preventDefault();
  showToast('📋 Blood request submitted successfully!', 3000);
  e.target.reset();
}

// ── Emergency Actions ──────────────────────────────────────
function currentTime() {
  return new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
}

function notifyDonors(btn) {
  btn.textContent = '✓ Notified';
  btn.disabled = true;
  btn.style.background = 'var(--green-light)';
  btn.style.color = 'var(--green)';
  btn.style.borderColor = 'var(--green)';

  addNotifLog(`${currentTime()} — SMS & push notifications sent to nearby matching donors`);
  showToast('📢 Donors notified successfully!', 2500);
}

function notifyAllDonors() {
  addNotifLog(`${currentTime()} — Broadcast alert sent to ALL available donors in the system`);
  showToast('🚨 All donors have been notified of emergency requests!', 3000);
}

function addNotifLog(message) {
  const log = document.getElementById('notif-log');
  if (!log) return;

  const item = document.createElement('div');
  item.className = 'notif-log-item';
  item.innerHTML = `
    <span class="notif-log-time">${currentTime()}</span>
    <span class="notif-log-msg">${message}</span>
  `;

  log.insertBefore(item, log.firstChild);
}

// ── Logout ─────────────────────────────────────────────────
function logout() {
  document.getElementById('main-layout').classList.add('hidden');
  document.getElementById('main-layout').style.display = 'none';
  document.getElementById('page-login').classList.add('active');

  // Reset charts so they re-init on next login
  Object.values(charts).forEach(c => c.destroy());
  charts = {};
}

// ── Toast ──────────────────────────────────────────────────
let toastTimer = null;

function showToast(message, duration = 3000) {
  const toast = document.getElementById('toast');
  if (!toast) return;

  toast.textContent = message;
  toast.classList.remove('hidden');

  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.add('hidden'), duration);
}
