/* =========================================================
   LearnFlow — Interactive Learning Platform
   app.js
   ========================================================= */

'use strict';

// =========================================================
// DATA
// =========================================================

const COURSES = [
  {
    id: 1, title: 'Python for Data Science', instructor: 'Dr. Sarah Chen',
    category: 'data-science', emoji: '🐍',
    gradient: 'linear-gradient(135deg,#6366f1,#8b5cf6)',
    rating: 4.9, students: 18200, lessons: 24, hours: '12h 30m',
    price: 'Free', enrolled: true, progress: 50,
    badge: 'Bestseller',
    description: 'Master Python for data analysis using NumPy, Pandas, Matplotlib, and Scikit-learn.',
    modules: [
      { title: 'Getting Started', lessons: [
        { id: 101, title: 'Introduction to Python', duration: '8:22', done: true },
        { id: 102, title: 'Setting up Jupyter', duration: '5:10', done: true },
        { id: 103, title: 'Variables & Data Types', duration: '12:00', done: true },
      ]},
      { title: 'NumPy Basics', lessons: [
        { id: 104, title: 'Arrays and Operations', duration: '14:15', done: true },
        { id: 105, title: 'Indexing & Slicing', duration: '10:30', done: false },
        { id: 106, title: 'Broadcasting', duration: '9:45', done: false },
      ]},
      { title: 'Pandas', lessons: [
        { id: 107, title: 'DataFrames', duration: '16:00', done: false },
        { id: 108, title: 'Merging & Grouping', duration: '18:20', done: false },
      ]},
    ],
    currentLesson: 105,
  },
  {
    id: 2, title: 'React & Modern JavaScript', instructor: 'Alex Johnson',
    category: 'programming', emoji: '⚛️',
    gradient: 'linear-gradient(135deg,#f59e0b,#ef4444)',
    rating: 4.8, students: 24500, lessons: 30, hours: '18h',
    price: '$49', enrolled: true, progress: 27,
    badge: 'Hot',
    description: 'Build modern web apps with React 18, Hooks, Context API, and Redux Toolkit.',
    modules: [
      { title: 'JS Essentials', lessons: [
        { id: 201, title: 'ES6+ Features', duration: '15:00', done: true },
        { id: 202, title: 'Arrow Functions & Closures', duration: '11:30', done: true },
      ]},
      { title: 'React Fundamentals', lessons: [
        { id: 203, title: 'JSX & Components', duration: '13:20', done: true },
        { id: 204, title: 'Props & State', duration: '12:45', done: false },
      ]},
      { title: 'Hooks Deep Dive', lessons: [
        { id: 205, title: 'useState & useEffect', duration: '20:00', done: false },
        { id: 206, title: 'useContext & useRef', duration: '14:10', done: false },
        { id: 207, title: 'Custom Hooks', duration: '16:30', done: false },
      ]},
    ],
    currentLesson: 204,
  },
  {
    id: 3, title: 'SQL Mastery', instructor: 'Maria Santos',
    category: 'data-science', emoji: '🗄️',
    gradient: 'linear-gradient(135deg,#10b981,#059669)',
    rating: 4.7, students: 11300, lessons: 20, hours: '8h',
    price: 'Free', enrolled: true, progress: 90,
    badge: null,
    description: 'From basic queries to advanced window functions, CTEs, and query optimization.',
    modules: [
      { title: 'SQL Basics', lessons: [
        { id: 301, title: 'SELECT & WHERE', duration: '7:00', done: true },
        { id: 302, title: 'JOINs Explained', duration: '9:30', done: true },
      ]},
      { title: 'Advanced SQL', lessons: [
        { id: 303, title: 'Subqueries & CTEs', duration: '12:00', done: true },
        { id: 304, title: 'Window Functions', duration: '15:00', done: false },
        { id: 305, title: 'Query Optimization', duration: '11:20', done: false },
      ]},
    ],
    currentLesson: 304,
  },
  {
    id: 4, title: 'UI/UX Design Fundamentals', instructor: 'Priya Sharma',
    category: 'design', emoji: '🎨',
    gradient: 'linear-gradient(135deg,#ec4899,#f43f5e)',
    rating: 4.9, students: 9800, lessons: 22, hours: '10h',
    price: '$29', enrolled: false, progress: 0,
    badge: 'New',
    description: 'Learn Figma, design systems, wireframing, and prototyping for beautiful UIs.',
    modules: [],
    currentLesson: null,
  },
  {
    id: 5, title: 'Docker & Kubernetes', instructor: 'Tom Wei',
    category: 'devops', emoji: '🐳',
    gradient: 'linear-gradient(135deg,#0ea5e9,#2563eb)',
    rating: 4.6, students: 7600, lessons: 28, hours: '14h',
    price: '$39', enrolled: false, progress: 0,
    badge: null,
    description: 'Containerize and orchestrate applications using Docker, Kubernetes, and Helm.',
    modules: [],
    currentLesson: null,
  },
  {
    id: 6, title: 'Machine Learning A–Z', instructor: 'Dr. Aisha Patel',
    category: 'data-science', emoji: '🤖',
    gradient: 'linear-gradient(135deg,#f97316,#ef4444)',
    rating: 4.8, students: 32100, lessons: 42, hours: '26h',
    price: '$59', enrolled: false, progress: 0,
    badge: 'Bestseller',
    description: 'Supervised, unsupervised learning and deep neural networks — all from scratch.',
    modules: [],
    currentLesson: null,
  },
  {
    id: 7, title: 'Business Analytics', instructor: 'James Miller',
    category: 'business', emoji: '📊',
    gradient: 'linear-gradient(135deg,#14b8a6,#0d9488)',
    rating: 4.5, students: 5400, lessons: 18, hours: '9h',
    price: '$19', enrolled: false, progress: 0,
    badge: 'New',
    description: 'Data-driven decision making with Excel, Power BI and strategic thinking.',
    modules: [],
    currentLesson: null,
  },
  {
    id: 8, title: 'Node.js Backend Development', instructor: 'Carlos Rivera',
    category: 'programming', emoji: '🟢',
    gradient: 'linear-gradient(135deg,#22c55e,#15803d)',
    rating: 4.7, students: 13200, lessons: 34, hours: '20h',
    price: '$44', enrolled: false, progress: 0,
    badge: null,
    description: 'REST APIs, authentication, databases, and deployment with Node.js and Express.',
    modules: [],
    currentLesson: null,
  },
];

const CERTIFICATES = [
  { title: 'HTML & CSS Foundations', date: 'Jan 12, 2025', emoji: '🌐', gradient: 'linear-gradient(135deg,#6366f1,#8b5cf6)' },
  { title: 'JavaScript Essentials', date: 'Feb 3, 2025',  emoji: '🟡', gradient: 'linear-gradient(135deg,#f59e0b,#ef4444)' },
  { title: 'Git & Version Control',  date: 'Feb 28, 2025', emoji: '🔀', gradient: 'linear-gradient(135deg,#10b981,#059669)' },
];

// =========================================================
// STATE
// =========================================================
const state = {
  currentView: 'dashboard',
  previousView: 'dashboard',
  currentCourse: null,
  currentLessonId: null,
  categoryFilter: 'all',
  searchQuery: '',
  myLearningTab: 'in-progress',
  lessonTab: 'overview',
  videoPlaying: false,
  videoProgress: 0,
  enrolledCourses: new Set(COURSES.filter(c => c.enrolled).map(c => c.id)),
};

// =========================================================
// DOM HELPERS
// =========================================================
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

function toast(message, duration = 3000) {
  const el = $('#toast');
  el.textContent = message;
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), duration);
}

// =========================================================
// NAVIGATION / ROUTING
// =========================================================
function navigate(view) {
  $$('.view').forEach(v => v.classList.remove('active'));
  $$('.nav-item').forEach(n => n.classList.remove('active'));

  const target = $(`#view-${view}`);
  if (target) target.classList.add('active');

  const navItem = $(`.nav-item[data-view="${view}"]`);
  if (navItem) navItem.classList.add('active');

  state.previousView = state.currentView;
  state.currentView = view;

  // Render dynamic content per view
  if (view === 'dashboard') renderDashboard();
  if (view === 'courses') renderBrowse();
  if (view === 'my-learning') renderMyLearning();
  if (view === 'progress') renderProgress();
  if (view === 'certificates') renderCertificates();

  // Close sidebar on mobile
  if (window.innerWidth <= 768) {
    $('#sidebar').classList.remove('open');
  }
}

// =========================================================
// COURSE CARD RENDERER
// =========================================================
function courseCardHTML(course, showProgress = false) {
  const enrolled = state.enrolledCourses.has(course.id);
  return `
    <div class="course-card" data-course-id="${course.id}">
      <div class="course-card-thumb" style="background:${course.gradient}">
        ${course.badge ? `<span class="course-card-badge">${course.badge}</span>` : ''}
        <span>${course.emoji}</span>
      </div>
      <div class="course-card-body">
        <div class="course-card-title">${course.title}</div>
        <div class="course-card-instructor">${course.instructor}</div>
        <div class="course-card-meta">
          <span class="rating">⭐ ${course.rating}</span>
          <span>${course.students.toLocaleString()} students</span>
          <span>${course.lessons} lessons</span>
        </div>
        ${showProgress && enrolled ? `
          <div class="progress-bar" style="margin-bottom:.5rem">
            <div class="progress-fill" style="width:${course.progress}%"></div>
          </div>
        ` : ''}
        <div class="course-card-footer">
          <span class="course-price">${course.price}</span>
          <button class="btn-enroll ${enrolled ? 'enrolled' : ''}" data-enroll="${course.id}">
            ${enrolled ? '▶ Continue' : 'Enroll Now'}
          </button>
        </div>
      </div>
    </div>
  `;
}

// =========================================================
// DASHBOARD VIEW
// =========================================================
function renderDashboard() {
  const recommended = COURSES.filter(c => !state.enrolledCourses.has(c.id)).slice(0, 4);
  const grid = $('#recommended-grid');
  if (grid) {
    grid.innerHTML = recommended.map(c => courseCardHTML(c)).join('');
    attachCourseCardListeners(grid);
  }
}

// =========================================================
// BROWSE COURSES VIEW
// =========================================================
function renderBrowse() {
  let filtered = [...COURSES];

  if (state.categoryFilter !== 'all') {
    filtered = filtered.filter(c => c.category === state.categoryFilter);
  }
  if (state.searchQuery) {
    const q = state.searchQuery.toLowerCase();
    filtered = filtered.filter(c =>
      c.title.toLowerCase().includes(q) ||
      c.instructor.toLowerCase().includes(q) ||
      c.category.toLowerCase().includes(q)
    );
  }

  // Apply sort
  const sortVal = $('#sort-select')?.value || 'popular';
  if (sortVal === 'popular') {
    filtered.sort((a, b) => b.students - a.students);
  } else if (sortVal === 'rating') {
    filtered.sort((a, b) => b.rating - a.rating);
  } else if (sortVal === 'newest') {
    // Reverse catalogue order to simulate newest-first
    filtered.reverse();
  }

  const countEl = $('#course-count');
  if (countEl) countEl.textContent = filtered.length;

  const grid = $('#browse-grid');
  if (!grid) return;

  if (filtered.length === 0) {
    grid.innerHTML = `
      <div class="empty-state" style="grid-column:1/-1">
        <div class="empty-icon">🔍</div>
        <p>No courses found for "<strong>${state.searchQuery}</strong>"</p>
      </div>
    `;
    return;
  }

  grid.innerHTML = filtered.map(c => courseCardHTML(c, true)).join('');
  attachCourseCardListeners(grid);
}

// =========================================================
// MY LEARNING VIEW
// =========================================================
function renderMyLearning() {
  const tab = state.myLearningTab;
  let courses;

  if (tab === 'in-progress') {
    courses = COURSES.filter(c => state.enrolledCourses.has(c.id) && c.progress < 100 && c.progress > 0);
  } else if (tab === 'completed') {
    courses = COURSES.filter(c => state.enrolledCourses.has(c.id) && c.progress === 100);
  } else {
    courses = []; // bookmarked — empty for demo
  }

  const list = $('#my-course-list');
  if (!list) return;

  if (courses.length === 0) {
    list.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">${tab === 'completed' ? '🏆' : tab === 'bookmarked' ? '🔖' : '📚'}</div>
        <p>${tab === 'completed' ? 'No completed courses yet — keep going!' :
             tab === 'bookmarked' ? 'No bookmarked courses yet.' :
             'No courses in progress.'}</p>
      </div>
    `;
    return;
  }

  list.innerHTML = courses.map(c => `
    <div class="my-course-item" data-course-id="${c.id}">
      <div class="my-course-thumb" style="background:${c.gradient}">
        ${c.emoji}
      </div>
      <div class="my-course-info">
        <h3>${c.title}</h3>
        <p>${c.instructor} · ${c.hours} · ${c.lessons} lessons</p>
        <div class="progress-bar">
          <div class="progress-fill" style="width:${c.progress}%"></div>
        </div>
      </div>
      <div class="my-course-actions">
        <span style="font-size:.85rem;color:var(--text-muted)">${c.progress}%</span>
        <button class="btn-primary" data-course-id="${c.id}">Continue</button>
      </div>
    </div>
  `).join('');

  list.querySelectorAll('.my-course-item, .btn-primary').forEach(el => {
    el.addEventListener('click', () => {
      const cid = parseInt(el.closest('[data-course-id]').dataset.courseId, 10);
      openLesson(cid);
    });
  });
}

// =========================================================
// LESSON PLAYER
// =========================================================
function openLesson(courseId, lessonId) {
  const course = COURSES.find(c => c.id === courseId);
  if (!course || !course.modules.length) {
    toast('Lesson content coming soon!');
    return;
  }

  // Resolve lesson
  const allLessons = course.modules.flatMap(m => m.lessons);
  const lesson = lessonId
    ? allLessons.find(l => l.id === lessonId)
    : allLessons.find(l => l.id === course.currentLesson) || allLessons[0];

  if (!lesson) { toast('Lesson not found.'); return; }

  state.currentCourse = course;
  state.currentLessonId = lesson.id;
  state.videoProgress = 0;
  state.videoPlaying = false;

  // Update UI
  $('#lesson-title-display').textContent = lesson.title;
  $('#lesson-heading').textContent = lesson.title;
  $('#lesson-desc').textContent = course.description;
  $('#course-title-sidebar').textContent = course.title;
  $('#vc-fill').style.width = '0%';
  $('#vc-time').textContent = `0:00 / ${lesson.duration}`;
  $('#vc-play').textContent = '▶ Play';

  renderCurriculum(course, lesson.id);

  navigate('lesson');
  resetLessonTabs();
}

function renderCurriculum(course, activeLessonId) {
  const list = $('#curriculum-list');
  list.innerHTML = course.modules.map((mod, idx) => `
    <div class="curriculum-module">
      <div class="module-header" data-mod="${idx}">
        <span>${mod.title}</span>
        <span>▾</span>
      </div>
      <div class="module-lessons ${idx === 0 || mod.lessons.some(l => l.id === activeLessonId) ? 'open' : ''}" id="mod-${idx}">
        ${mod.lessons.map(l => `
          <div class="curriculum-lesson ${l.done ? 'done' : ''} ${l.id === activeLessonId ? 'active' : ''}" data-lesson-id="${l.id}">
            <span class="lesson-icon">${l.done ? '✅' : '▶'}</span>
            <span>${l.title}</span>
            <span class="lesson-duration">${l.duration}</span>
          </div>
        `).join('')}
      </div>
    </div>
  `).join('');

  // Module toggle
  list.querySelectorAll('.module-header').forEach(h => {
    h.addEventListener('click', () => {
      const mod = $(`#mod-${h.dataset.mod}`);
      mod.classList.toggle('open');
    });
  });

  // Lesson click
  list.querySelectorAll('.curriculum-lesson').forEach(l => {
    l.addEventListener('click', () => {
      const lid = parseInt(l.dataset.lessonId, 10);
      openLesson(state.currentCourse.id, lid);
    });
  });
}

function resetLessonTabs() {
  $$('.lesson-tabs .tab').forEach(t => t.classList.remove('active'));
  const first = $('.lesson-tabs .tab');
  if (first) first.classList.add('active');
  state.lessonTab = 'overview';
  renderLessonTabContent('overview');
}

function renderLessonTabContent(tab) {
  const content = $('#lesson-tab-content');
  if (!content) return;
  const tabs = {
    overview: 'This lesson covers the fundamentals of the topic. Watch the video and complete the interactive exercises below to reinforce your understanding.',
    resources: '📄 <strong>Lesson Slides</strong> — Download PDF<br>📦 <strong>Starter Code</strong> — GitHub Repository<br>📖 <strong>Reading</strong> — Official Documentation Link',
    notes: '<em>Your notes will appear here. Start taking notes while watching the video.</em>',
    qa: '💬 <strong>3 questions</strong> from other students. Join the discussion!',
  };
  content.innerHTML = tabs[tab] || '';
}

// Video player simulation
let videoTimer = null;

function startVideo() {
  if (state.videoPlaying) return;
  state.videoPlaying = true;
  $('#vc-play').textContent = '⏸ Pause';
  videoTimer = setInterval(() => {
    state.videoProgress = Math.min(100, state.videoProgress + 0.5);
    $('#vc-fill').style.width = state.videoProgress + '%';
    const total = 12 * 60 + 30;
    const current = Math.round((state.videoProgress / 100) * total);
    const m = Math.floor(current / 60);
    const s = current % 60;
    $('#vc-time').textContent = `${m}:${String(s).padStart(2,'0')} / 12:30`;
    if (state.videoProgress >= 100) stopVideo();
  }, 200);
}

function stopVideo() {
  clearInterval(videoTimer);
  state.videoPlaying = false;
  $('#vc-play').textContent = '▶ Play';
}

// =========================================================
// PROGRESS VIEW
// =========================================================
function renderProgress() {
  const list = $('#progress-course-list');
  if (!list) return;
  const enrolled = COURSES.filter(c => state.enrolledCourses.has(c.id));
  list.innerHTML = enrolled.map(c => `
    <div class="progress-course-row">
      <span class="course-emoji">${c.emoji}</span>
      <span class="course-name">${c.title}</span>
      <div class="progress-bar">
        <div class="progress-fill" style="width:${c.progress}%"></div>
      </div>
      <span class="progress-pct">${c.progress}%</span>
    </div>
  `).join('');
}

// =========================================================
// CERTIFICATES VIEW
// =========================================================
function renderCertificates() {
  const grid = $('#cert-grid');
  if (!grid) return;
  if (CERTIFICATES.length === 0) {
    grid.innerHTML = '<div class="empty-state"><div class="empty-icon">🏆</div><p>Complete a course to earn your first certificate!</p></div>';
    return;
  }
  grid.innerHTML = CERTIFICATES.map(c => `
    <div class="cert-card" style="background:${c.gradient}">
      <div class="cert-icon">${c.emoji}</div>
      <div class="cert-title">${c.title}</div>
      <div class="cert-sub">Certificate of Completion · LearnFlow</div>
      <div class="cert-date">Issued: ${c.date}</div>
      <button class="cert-download">⬇ Download Certificate</button>
    </div>
  `).join('');

  grid.querySelectorAll('.cert-download').forEach((btn, i) => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      toast(`Downloading certificate: "${CERTIFICATES[i].title}"`);
    });
  });
}

// =========================================================
// EVENT DELEGATION — COURSE CARDS
// =========================================================
function attachCourseCardListeners(container) {
  container.querySelectorAll('.btn-enroll').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const courseId = parseInt(btn.dataset.enroll, 10);
      const course = COURSES.find(c => c.id === courseId);
      if (!course) return;

      if (state.enrolledCourses.has(courseId)) {
        openLesson(courseId);
      } else {
        state.enrolledCourses.add(courseId);
        course.enrolled = true;
        course.progress = 0;
        btn.textContent = '▶ Continue';
        btn.classList.add('enrolled');
        toast(`🎉 Enrolled in "${course.title}"!`);
      }
    });
  });

  container.querySelectorAll('.course-card').forEach(card => {
    card.addEventListener('click', (e) => {
      if (e.target.classList.contains('btn-enroll')) return;
      const courseId = parseInt(card.dataset.courseId, 10);
      if (state.enrolledCourses.has(courseId)) {
        openLesson(courseId);
      } else {
        // Show toast to enroll
        const c = COURSES.find(c => c.id === courseId);
        toast(`Click "Enroll Now" to start "${c?.title}"`);
      }
    });
  });
}

// =========================================================
// SEARCH
// =========================================================
function setupSearch() {
  const input = $('#search-input');
  if (!input) return;
  input.addEventListener('input', () => {
    state.searchQuery = input.value.trim();
    if (state.currentView !== 'courses') navigate('courses');
    else renderBrowse();
  });
}

// =========================================================
// CATEGORY FILTERS
// =========================================================
function setupFilters() {
  const filters = $('#category-filters');
  if (!filters) return;
  filters.addEventListener('click', (e) => {
    const chip = e.target.closest('.chip');
    if (!chip) return;
    $$('#category-filters .chip').forEach(c => c.classList.remove('active'));
    chip.classList.add('active');
    state.categoryFilter = chip.dataset.category;
    renderBrowse();
  });
}

// =========================================================
// SIDEBAR NAVIGATION
// =========================================================
function setupNav() {
  $$('.nav-item').forEach(item => {
    item.addEventListener('click', (e) => {
      e.preventDefault();
      navigate(item.dataset.view);
    });
  });

  // Link-all anchors in dashboard
  document.addEventListener('click', (e) => {
    const link = e.target.closest('.link-all');
    if (link) { e.preventDefault(); navigate(link.dataset.view); }

    // Continue card resume button
    const resume = e.target.closest('.btn-resume');
    if (resume) {
      const courseId = parseInt(resume.dataset.course, 10);
      openLesson(courseId);
    }

    // Continue card click
    const continueCard = e.target.closest('.continue-card');
    if (continueCard && !e.target.closest('.btn-resume')) {
      const courseId = parseInt(continueCard.dataset.course, 10);
      openLesson(courseId);
    }
  });
}

// =========================================================
// TABS — My Learning
// =========================================================
function setupTabs() {
  const tabsEl = $('#view-my-learning .tabs');
  if (tabsEl) {
    tabsEl.addEventListener('click', (e) => {
      const tab = e.target.closest('.tab');
      if (!tab) return;
      $$('#view-my-learning .tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      state.myLearningTab = tab.dataset.tab;
      renderMyLearning();
    });
  }

  // Lesson tabs
  const lessonTabsEl = $('.lesson-tabs');
  if (lessonTabsEl) {
    lessonTabsEl.addEventListener('click', (e) => {
      const tab = e.target.closest('.tab');
      if (!tab) return;
      $$('.lesson-tabs .tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      state.lessonTab = tab.dataset.lessonTab;
      renderLessonTabContent(state.lessonTab);
    });
  }
}

// =========================================================
// LESSON NAV BUTTONS
// =========================================================
function setupLessonNav() {
  $('#btn-back')?.addEventListener('click', () => {
    stopVideo();
    navigate(state.previousView || 'my-learning');
  });

  $('#btn-next-lesson')?.addEventListener('click', () => {
    if (!state.currentCourse) return;
    const allLessons = state.currentCourse.modules.flatMap(m => m.lessons);
    const idx = allLessons.findIndex(l => l.id === state.currentLessonId);
    if (idx < allLessons.length - 1) {
      // Mark current as done
      allLessons[idx].done = true;
      openLesson(state.currentCourse.id, allLessons[idx + 1].id);
    } else {
      toast('🎉 You have completed all lessons in this module!');
    }
  });

  $('#btn-prev-lesson')?.addEventListener('click', () => {
    if (!state.currentCourse) return;
    const allLessons = state.currentCourse.modules.flatMap(m => m.lessons);
    const idx = allLessons.findIndex(l => l.id === state.currentLessonId);
    if (idx > 0) {
      openLesson(state.currentCourse.id, allLessons[idx - 1].id);
    } else {
      toast('You are at the first lesson.');
    }
  });

  // Video play/pause
  $('#vc-play')?.addEventListener('click', () => {
    if (state.videoPlaying) stopVideo(); else startVideo();
  });

  // Click play icon in placeholder
  document.querySelector('.play-icon')?.addEventListener('click', startVideo);
}

// =========================================================
// MOBILE SIDEBAR TOGGLE
// =========================================================
function setupMobileToggle() {
  const toggle = $('#menu-toggle');
  const sidebar = $('#sidebar');
  toggle?.addEventListener('click', () => sidebar.classList.toggle('open'));

  // Close on outside click
  document.addEventListener('click', (e) => {
    if (window.innerWidth <= 768 &&
        sidebar.classList.contains('open') &&
        !sidebar.contains(e.target) &&
        e.target !== toggle) {
      sidebar.classList.remove('open');
    }
  });
}

// =========================================================
// SORT SELECT
// =========================================================
function setupSortSelect() {
  $('#sort-select')?.addEventListener('change', () => renderBrowse());
}

// =========================================================
// INIT
// =========================================================
function init() {
  setupNav();
  setupSearch();
  setupFilters();
  setupTabs();
  setupLessonNav();
  setupMobileToggle();
  setupSortSelect();
  renderDashboard();
  navigate('dashboard');
}

document.addEventListener('DOMContentLoaded', init);
