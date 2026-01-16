"""
Block Blast AI - Native Tkinter GUI

Provides a visual interface for training and watching the AI play.
"""
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import queue
import time
import sys
import io
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import redirect_stdout, redirect_stderr
import numpy as np

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


class OutputRedirector(io.StringIO):
    """Redirects stdout/stderr to a queue for GUI display."""

    def __init__(self, queue: queue.Queue, tag: str = "stdout"):
        super().__init__()
        self.queue = queue
        self.tag = tag

    def write(self, text):
        if text.strip():
            self.queue.put(('output', text))
        return len(text)

    def flush(self):
        pass


class BlockBlastGUI:
    """Main GUI application for Block Blast AI."""

    # Color scheme
    COLORS = {
        'bg': '#1a1a2e',
        'panel': '#16213e',
        'accent': '#e94560',
        'accent2': '#0f3460',
        'text': '#ffffff',
        'text_dim': '#8892b0',
        'terminal_bg': '#0d1117',
        'terminal_fg': '#c9d1d9',
        'grid_bg': '#0d1b2a',
        'empty_cell': '#1b263b',
        'piece_colors': [
            '#e94560', '#f39c12', '#9b59b6', '#3498db',
            '#2ecc71', '#1abc9c', '#e74c3c', '#f1c40f',
        ]
    }

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Block Blast AI")
        self.root.configure(bg=self.COLORS['bg'])
        self.root.geometry("1000x750")
        self.root.minsize(900, 650)

        # State
        self.is_training = False
        self.is_watching = False
        self.watch_speed = 500
        self.message_queue = queue.Queue()
        self.training_thread = None
        self.watch_thread = None

        # Game state
        self.board_state = np.zeros((8, 8), dtype=int)
        self.stats = {
            'score': 0, 'moves': 0, 'lines': 0,
            'combo': 0, 'games': 0, 'best_score': 0,
        }

        # Pages
        self.pages = {}
        self.current_page = None

        self._setup_styles()
        self._create_main_container()
        self._create_pages()
        self._show_page('menu')
        self._check_queue()

    def _setup_styles(self):
        """Configure ttk styles."""
        style = ttk.Style()
        style.theme_use('clam')

        style.configure('TFrame', background=self.COLORS['bg'])
        style.configure('Panel.TFrame', background=self.COLORS['panel'])

        style.configure(
            'Title.TLabel',
            background=self.COLORS['bg'],
            foreground=self.COLORS['accent'],
            font=('Segoe UI', 32, 'bold')
        )

        style.configure(
            'Subtitle.TLabel',
            background=self.COLORS['bg'],
            foreground=self.COLORS['text_dim'],
            font=('Segoe UI', 12)
        )

    def _create_main_container(self):
        """Create the main container for pages."""
        self.container = tk.Frame(self.root, bg=self.COLORS['bg'])
        self.container.pack(fill=tk.BOTH, expand=True)

    def _create_pages(self):
        """Create all pages."""
        self._create_menu_page()
        self._create_play_page()
        self._create_training_page()
        self._create_watch_page()

    def _show_page(self, page_name: str):
        """Show a specific page."""
        if self.current_page:
            self.pages[self.current_page].pack_forget()

        self.pages[page_name].pack(fill=tk.BOTH, expand=True)
        self.current_page = page_name

    # ==================== MENU PAGE ====================

    def _create_menu_page(self):
        """Create the main menu page."""
        page = tk.Frame(self.container, bg=self.COLORS['bg'])
        self.pages['menu'] = page

        # Center content
        center_frame = tk.Frame(page, bg=self.COLORS['bg'])
        center_frame.place(relx=0.5, rely=0.5, anchor='center')

        # Title
        title = tk.Label(
            center_frame,
            text="BLOCK BLAST AI",
            font=('Segoe UI', 42, 'bold'),
            bg=self.COLORS['bg'],
            fg=self.COLORS['accent']
        )
        title.pack(pady=(0, 10))

        # Subtitle
        subtitle = tk.Label(
            center_frame,
            text="Reinforcement Learning Agent with PPO",
            font=('Segoe UI', 14),
            bg=self.COLORS['bg'],
            fg=self.COLORS['text_dim']
        )
        subtitle.pack(pady=(0, 50))

        # Menu buttons
        btn_frame = tk.Frame(center_frame, bg=self.COLORS['bg'])
        btn_frame.pack()

        # Train button
        train_btn = tk.Button(
            btn_frame,
            text="Train AI",
            font=('Segoe UI', 16, 'bold'),
            bg=self.COLORS['accent'],
            fg=self.COLORS['text'],
            activebackground='#ff6b6b',
            activeforeground=self.COLORS['text'],
            width=20,
            height=2,
            cursor='hand2',
            relief='flat',
            command=lambda: self._show_page('training')
        )
        train_btn.pack(pady=10)

        # Watch button
        watch_btn = tk.Button(
            btn_frame,
            text="Watch AI Play",
            font=('Segoe UI', 16, 'bold'),
            bg=self.COLORS['accent2'],
            fg=self.COLORS['text'],
            activebackground='#1a5a9a',
            activeforeground=self.COLORS['text'],
            width=20,
            height=2,
            cursor='hand2',
            relief='flat',
            command=lambda: self._show_page('watch')
        )
        watch_btn.pack(pady=10)

        # Play button (human play)
        play_btn = tk.Button(
            btn_frame,
            text="Play Game",
            font=('Segoe UI', 16, 'bold'),
            bg='#2ecc71',
            fg=self.COLORS['text'],
            activebackground='#27ae60',
            activeforeground=self.COLORS['text'],
            width=20,
            height=2,
            cursor='hand2',
            relief='flat',
            command=self._start_human_play
        )
        play_btn.pack(pady=10)

        # Version info
        version_label = tk.Label(
            center_frame,
            text="Masked PPO with Action Masking",
            font=('Segoe UI', 10),
            bg=self.COLORS['bg'],
            fg=self.COLORS['text_dim']
        )
        version_label.pack(pady=(40, 0))

    # ==================== HUMAN PLAY PAGE ====================

    def _create_play_page(self):
        """Create the human play page with drag-and-drop."""
        from game.engine import GameEngine
        
        page = tk.Frame(self.container, bg=self.COLORS['bg'])
        self.pages['play'] = page

        # Header
        header = tk.Frame(page, bg=self.COLORS['panel'], height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        back_btn = tk.Button(
            header,
            text="< Back",
            font=('Segoe UI', 11),
            bg=self.COLORS['accent2'],
            fg=self.COLORS['text'],
            activebackground='#1a5a9a',
            relief='flat',
            cursor='hand2',
            command=self._back_from_play
        )
        back_btn.pack(side=tk.LEFT, padx=15, pady=12)

        title = tk.Label(
            header,
            text="Play Block Blast",
            font=('Segoe UI', 18, 'bold'),
            bg=self.COLORS['panel'],
            fg=self.COLORS['text']
        )
        title.pack(side=tk.LEFT, padx=20)

        # Main content
        content = tk.Frame(page, bg=self.COLORS['bg'])
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Left side - Game board
        left_panel = tk.Frame(content, bg=self.COLORS['panel'])
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20))

        board_container = tk.Frame(left_panel, bg=self.COLORS['panel'])
        board_container.pack(expand=True)

        self.play_cell_size = 55
        board_size = self.play_cell_size * 8 + 2

        self.play_board_canvas = tk.Canvas(
            board_container,
            width=board_size,
            height=board_size,
            bg=self.COLORS['grid_bg'],
            highlightthickness=3,
            highlightbackground='#2ecc71'
        )
        self.play_board_canvas.pack(pady=20)
        
        # Bind board events
        self.play_board_canvas.bind('<Motion>', self._on_board_motion)
        self.play_board_canvas.bind('<Button-1>', self._on_board_click)
        self.play_board_canvas.bind('<Leave>', self._on_board_leave)

        # Pieces tray
        pieces_label = tk.Label(
            board_container,
            text="Drag a piece to place it",
            font=('Segoe UI', 12, 'bold'),
            bg=self.COLORS['panel'],
            fg=self.COLORS['text']
        )
        pieces_label.pack(pady=(10, 5))

        self.play_pieces_frame = tk.Frame(board_container, bg=self.COLORS['panel'])
        self.play_pieces_frame.pack(pady=10)

        self.play_piece_canvases = []
        for i in range(3):
            canvas = tk.Canvas(
                self.play_pieces_frame,
                width=100,
                height=100,
                bg=self.COLORS['grid_bg'],
                highlightthickness=2,
                highlightbackground=self.COLORS['accent2'],
                cursor='hand2'
            )
            canvas.pack(side=tk.LEFT, padx=10)
            canvas.bind('<Button-1>', lambda e, idx=i: self._on_piece_click(idx))
            canvas.bind('<B1-Motion>', lambda e, idx=i: self._on_piece_drag(e, idx))
            canvas.bind('<ButtonRelease-1>', lambda e, idx=i: self._on_piece_release(e, idx))
            self.play_piece_canvases.append(canvas)

        # Right side - Stats panel
        right_panel = tk.Frame(content, bg=self.COLORS['panel'], width=280)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel.pack_propagate(False)

        stats_inner = tk.Frame(right_panel, bg=self.COLORS['panel'])
        stats_inner.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Score display
        score_title = tk.Label(
            stats_inner,
            text="SCORE",
            font=('Segoe UI', 12),
            bg=self.COLORS['panel'],
            fg=self.COLORS['text_dim']
        )
        score_title.pack(pady=(0, 5))

        self.play_score_label = tk.Label(
            stats_inner,
            text="0",
            font=('Segoe UI', 36, 'bold'),
            bg=self.COLORS['panel'],
            fg='#2ecc71'
        )
        self.play_score_label.pack(pady=(0, 20))

        # Stats
        self.play_stat_labels = {}
        stats_items = [
            ('moves', 'Moves'),
            ('lines', 'Lines Cleared'),
            ('combo', 'Current Streak'),
            ('max_combo', 'Best Streak'),
        ]

        for key, label_text in stats_items:
            frame = tk.Frame(stats_inner, bg=self.COLORS['panel'])
            frame.pack(fill=tk.X, pady=5)

            label = tk.Label(
                frame,
                text=f"{label_text}:",
                font=('Segoe UI', 11),
                bg=self.COLORS['panel'],
                fg=self.COLORS['text_dim'],
                anchor='w'
            )
            label.pack(side=tk.LEFT)

            value = tk.Label(
                frame,
                text="0",
                font=('Segoe UI', 12, 'bold'),
                bg=self.COLORS['panel'],
                fg=self.COLORS['accent'],
                anchor='e'
            )
            value.pack(side=tk.RIGHT)
            self.play_stat_labels[key] = value

        # New Game button
        self.new_game_btn = tk.Button(
            stats_inner,
            text="New Game",
            font=('Segoe UI', 12, 'bold'),
            bg='#2ecc71',
            fg=self.COLORS['text'],
            activebackground='#27ae60',
            width=16,
            height=2,
            cursor='hand2',
            relief='flat',
            command=self._new_game
        )
        self.new_game_btn.pack(pady=(30, 10))

        # Status
        self.play_status = tk.Label(
            stats_inner,
            text="Click a piece, then click on the board",
            font=('Segoe UI', 10),
            bg=self.COLORS['panel'],
            fg=self.COLORS['text_dim'],
            wraplength=200
        )
        self.play_status.pack(pady=(10, 0))
        
        # Game Over overlay (hidden by default)
        self.game_over_frame = tk.Frame(page, bg='#000000')
        self.game_over_label = tk.Label(
            self.game_over_frame,
            text="GAME OVER",
            font=('Segoe UI', 32, 'bold'),
            bg='#000000',
            fg=self.COLORS['accent']
        )
        self.game_over_label.pack(pady=(20, 10))
        
        self.final_score_label = tk.Label(
            self.game_over_frame,
            text="Score: 0",
            font=('Segoe UI', 18),
            bg='#000000',
            fg=self.COLORS['text']
        )
        self.final_score_label.pack(pady=(0, 20))
        
        restart_btn = tk.Button(
            self.game_over_frame,
            text="Play Again",
            font=('Segoe UI', 14, 'bold'),
            bg='#2ecc71',
            fg=self.COLORS['text'],
            width=15,
            height=2,
            relief='flat',
            cursor='hand2',
            command=self._new_game
        )
        restart_btn.pack()

        # Play state
        self.play_engine = None
        self.selected_piece_idx = None
        self.dragging = False
        self.ghost_items = []
        
        # Floating piece window for drag visualization
        self.drag_window = None

    def _start_human_play(self):
        """Start human play mode."""
        self._show_page('play')
        self._new_game()

    def _back_from_play(self):
        """Go back from play page."""
        self.game_over_frame.place_forget()
        self._show_page('menu')

    def _new_game(self):
        """Start a new game."""
        from game.engine import GameEngine
        
        self.play_engine = GameEngine()
        self.selected_piece_idx = None
        # Track colors for each cell (engine only stores 1/0)
        self.board_colors = [[None for _ in range(8)] for _ in range(8)]
        self.game_over_frame.place_forget()
        self._update_play_display()
        self.play_status.config(text="Drag a piece onto the board!")

    def _update_play_display(self):
        """Update the play page display."""
        if not self.play_engine:
            return

        # Update board
        self._draw_play_board()
        
        # Update pieces
        self._draw_play_pieces()
        
        # Update stats
        self.play_score_label.config(text=f"{self.play_engine.score:,}")
        stats = self.play_engine.get_statistics()
        self.play_stat_labels['moves'].config(text=str(self.play_engine.moves_made))
        self.play_stat_labels['lines'].config(text=str(self.play_engine.total_lines_cleared))
        self.play_stat_labels['combo'].config(text=f"x{self.play_engine.combo_count + 1}")
        self.play_stat_labels['max_combo'].config(text=f"x{self.play_engine.max_combo + 1}")

    def _draw_play_board(self):
        """Draw the game board for human play."""
        self.play_board_canvas.delete('all')
        
        if not self.play_engine:
            return

        board = self.play_engine.board.grid

        for row in range(8):
            for col in range(8):
                x1 = col * self.play_cell_size + 1
                y1 = row * self.play_cell_size + 1
                x2 = x1 + self.play_cell_size - 2
                y2 = y1 + self.play_cell_size - 2

                if board[row, col]:
                    # Use tracked color or fallback
                    color = self.board_colors[row][col] if self.board_colors[row][col] else self.COLORS['piece_colors'][0]
                    self._draw_3d_block(self.play_board_canvas, x1, y1, x2, y2, color)
                else:
                    self.play_board_canvas.create_rectangle(
                        x1, y1, x2, y2,
                        fill=self.COLORS['empty_cell'],
                        outline='#2d3748',
                        width=1
                    )

    def _draw_3d_block(self, canvas, x1, y1, x2, y2, color):
        """Draw a 3D-style block with shading."""
        # Main fill
        canvas.create_rectangle(
            x1, y1, x2, y2,
            fill=color,
            outline='',
            width=0
        )
        # Top highlight
        canvas.create_polygon(
            x1, y1, x2, y1, x2 - 4, y1 + 4, x1 + 4, y1 + 4,
            fill=self._lighten_color(color, 0.35),
            outline=''
        )
        # Left highlight
        canvas.create_polygon(
            x1, y1, x1 + 4, y1 + 4, x1 + 4, y2 - 4, x1, y2,
            fill=self._lighten_color(color, 0.2),
            outline=''
        )
        # Bottom shadow
        canvas.create_polygon(
            x1, y2, x1 + 4, y2 - 4, x2 - 4, y2 - 4, x2, y2,
            fill=self._darken_color(color, 0.3),
            outline=''
        )
        # Right shadow
        canvas.create_polygon(
            x2, y1, x2, y2, x2 - 4, y2 - 4, x2 - 4, y1 + 4,
            fill=self._darken_color(color, 0.2),
            outline=''
        )
        # Inner shine
        canvas.create_rectangle(
            x1 + 6, y1 + 6, x1 + 14, y1 + 10,
            fill=self._lighten_color(color, 0.5),
            outline=''
        )

    def _draw_play_pieces(self):
        """Draw available pieces for human play."""
        if not self.play_engine:
            return

        for i, canvas in enumerate(self.play_piece_canvases):
            canvas.delete('all')
            
            # Highlight selected piece
            if i == self.selected_piece_idx:
                canvas.config(highlightbackground='#2ecc71', highlightthickness=3)
            else:
                canvas.config(highlightbackground=self.COLORS['accent2'], highlightthickness=2)

            if i < len(self.play_engine.current_pieces) and not self.play_engine.pieces_used[i]:
                piece = self.play_engine.current_pieces[i]
                shape = piece.get_shape_array()
                cell_size = 20
                
                h, w = shape.shape
                offset_x = (100 - w * cell_size) // 2
                offset_y = (100 - h * cell_size) // 2

                color = self.COLORS['piece_colors'][i % len(self.COLORS['piece_colors'])]

                for row in range(h):
                    for col in range(w):
                        if shape[row, col]:
                            x1 = offset_x + col * cell_size
                            y1 = offset_y + row * cell_size
                            x2 = x1 + cell_size - 2
                            y2 = y1 + cell_size - 2

                            # Use 3D block style for piece preview
                            self._draw_3d_block(canvas, x1, y1, x2, y2, color)
            else:
                # Piece already used - show checkmark or empty
                canvas.create_text(
                    50, 50,
                    text="✓" if self.play_engine.pieces_used[i] else "",
                    font=('Segoe UI', 24),
                    fill=self.COLORS['text_dim']
                )

    def _on_piece_click(self, piece_idx: int):
        """Handle piece click - start dragging."""
        if not self.play_engine or self.play_engine.is_game_over():
            return
        if self.play_engine.pieces_used[piece_idx]:
            return
            
        self.selected_piece_idx = piece_idx
        self.dragging = True
        self._draw_play_pieces()
        self.play_status.config(text=f"Drag to board and release to place")
        
        # Create floating piece window
        self._create_drag_window()
        
        # Start tracking mouse globally
        self.root.bind('<Motion>', self._on_global_motion)
        self.root.bind('<ButtonRelease-1>', self._on_global_release)
        
        # Show floating piece at current mouse position
        self._update_drag_position()

    def _create_drag_window(self):
        """Create a floating window showing the dragged piece."""
        if self.drag_window:
            self.drag_window.destroy()
        
        if self.selected_piece_idx is None or not self.play_engine:
            return
        
        piece = self.play_engine.current_pieces[self.selected_piece_idx]
        shape = piece.get_shape_array()
        cell_size = self.play_cell_size  # Same size as board cells
        
        h, w = shape.shape
        win_w = w * cell_size
        win_h = h * cell_size
        
        # Create toplevel window with transparent background
        self.drag_window = tk.Toplevel(self.root)
        self.drag_window.overrideredirect(True)  # No window decorations
        self.drag_window.attributes('-topmost', True)
        self.drag_window.attributes('-alpha', 0.9)
        # Use a key color for transparency on Windows
        self.drag_window.attributes('-transparentcolor', '#010101')
        self.drag_window.configure(bg='#010101')  # This becomes transparent
        
        # Create canvas for piece with transparent background
        canvas = tk.Canvas(
            self.drag_window,
            width=win_w,
            height=win_h,
            bg='#010101',  # Transparent
            highlightthickness=0
        )
        canvas.pack()
        
        color = self.COLORS['piece_colors'][self.selected_piece_idx % len(self.COLORS['piece_colors'])]
        
        for row in range(h):
            for col in range(w):
                if shape[row, col]:
                    x1 = col * cell_size
                    y1 = row * cell_size
                    x2 = x1 + cell_size - 2
                    y2 = y1 + cell_size - 2
                    
                    # Main block
                    canvas.create_rectangle(
                        x1, y1, x2, y2,
                        fill=color,
                        outline=self._lighten_color(color),
                        width=2
                    )
                    # Highlight shine effect
                    canvas.create_rectangle(
                        x1 + 4, y1 + 4, x2 - 4, y1 + 12,
                        fill=self._lighten_color(color, 0.3),
                        outline=''
                    )
        
        # Store piece dimensions for position calculation
        self.drag_piece_width = w
        self.drag_piece_height = h
        
        # Offset from cursor - center the piece on cursor
        self.drag_window_offset = (win_w // 2, win_h // 2)

    def _destroy_drag_window(self):
        """Destroy the floating drag window."""
        if self.drag_window:
            self.drag_window.destroy()
            self.drag_window = None

    def _on_piece_drag(self, event, piece_idx: int):
        """Handle piece drag motion (backup for within-canvas drag)."""
        if not self.dragging:
            return
        self._update_drag_position()

    def _on_piece_release(self, event, piece_idx: int):
        """Handle piece release within piece canvas."""
        # Let global release handler deal with it
        pass

    def _update_drag_position(self):
        """Update floating piece and ghost based on current mouse position."""
        if not self.dragging or self.selected_piece_idx is None:
            return
        
        # Get mouse position
        mouse_x = self.root.winfo_pointerx()
        mouse_y = self.root.winfo_pointery()
        
        # Move floating window centered on cursor
        if self.drag_window:
            offset_x, offset_y = self.drag_window_offset
            self.drag_window.geometry(f'+{mouse_x - offset_x}+{mouse_y - offset_y}')
        
        # Get board position
        board_x = self.play_board_canvas.winfo_rootx()
        board_y = self.play_board_canvas.winfo_rooty()
        
        # Calculate where the PIECE VISUAL's top-left is, not the cursor
        piece_left = mouse_x - self.drag_window_offset[0]
        piece_top = mouse_y - self.drag_window_offset[1]
        
        # Calculate grid position based on piece visual's center
        piece_center_x = piece_left + (self.drag_piece_width * self.play_cell_size) // 2
        piece_center_y = piece_top + (self.drag_piece_height * self.play_cell_size) // 2
        
        # Grid position from piece center
        rel_x = piece_center_x - board_x
        rel_y = piece_center_y - board_y
        
        grid_col = int(rel_x / self.play_cell_size) - self.drag_piece_width // 2
        grid_row = int(rel_y / self.play_cell_size) - self.drag_piece_height // 2
        
        # Store grid position for placement
        self.drag_row = grid_row
        self.drag_col = grid_col
        
        self._show_ghost_piece(grid_row, grid_col)

    def _on_global_motion(self, event):
        """Handle mouse motion during drag."""
        if not self.dragging:
            return
        self._update_drag_position()

    def _on_global_release(self, event):
        """Handle mouse release anywhere."""
        if not self.dragging:
            return
        
        # Unbind global events
        self.root.unbind('<Motion>')
        self.root.unbind('<ButtonRelease-1>')
        
        # Destroy floating piece
        self._destroy_drag_window()
        
        self._clear_ghost()
        
        # Get drop position relative to board
        board_x = self.play_board_canvas.winfo_rootx()
        board_y = self.play_board_canvas.winfo_rooty()
        board_w = self.play_board_canvas.winfo_width()
        board_h = self.play_board_canvas.winfo_height()
        
        mouse_x = event.x_root - board_x
        mouse_y = event.y_root - board_y
        
        # Check if dropped on board
        if 0 <= mouse_x < board_w and 0 <= mouse_y < board_h:
            # Use stored position from last ghost preview (most accurate)
            row = getattr(self, 'drag_row', int(mouse_y / self.play_cell_size))
            col = getattr(self, 'drag_col', int(mouse_x / self.play_cell_size))
            
            self._try_place_piece(row, col)
        else:
            self.play_status.config(text="Dropped outside board - try again!")
        
        self.dragging = False
        self.selected_piece_idx = None
        self._draw_play_pieces()

    def _on_board_motion(self, event):
        """Handle mouse motion over board."""
        if self.selected_piece_idx is not None:
            col = int(event.x // self.play_cell_size)
            row = int(event.y // self.play_cell_size)
            self._show_ghost_piece(row, col)

    def _on_board_click(self, event):
        """Handle click on board."""
        if self.selected_piece_idx is None:
            return
        
        col = int(event.x // self.play_cell_size)
        row = int(event.y // self.play_cell_size)
        self._try_place_piece(row, col)
        self._clear_ghost()
        self.selected_piece_idx = None
        self._draw_play_pieces()

    def _on_board_leave(self, event):
        """Handle mouse leaving board."""
        self._clear_ghost()

    def _show_ghost_piece(self, row: int, col: int):
        """Show ghost preview of piece placement - only when valid."""
        self._clear_ghost()
        
        if self.selected_piece_idx is None or not self.play_engine:
            return
        
        if self.play_engine.pieces_used[self.selected_piece_idx]:
            return
        
        piece = self.play_engine.current_pieces[self.selected_piece_idx]
        
        # Only show ghost if placement is valid (not overlapping or out of bounds)
        can_place = self.play_engine.can_place_piece(self.selected_piece_idx, row, col)
        
        if not can_place:
            # Don't show ghost for invalid placements
            return
        
        # Valid placement - show ghost with piece color but darker/semi-transparent look
        color = self.COLORS['piece_colors'][self.selected_piece_idx % len(self.COLORS['piece_colors'])]
        # Create a darker version for ghost
        ghost_color = self._darken_color(color, 0.5)
        
        for r, c in piece.blocks:
            cell_row = row + r
            cell_col = col + c
            if 0 <= cell_row < 8 and 0 <= cell_col < 8:
                x1 = cell_col * self.play_cell_size + 1
                y1 = cell_row * self.play_cell_size + 1
                x2 = x1 + self.play_cell_size - 2
                y2 = y1 + self.play_cell_size - 2
                
                # Ghost block with stipple for transparency effect
                item = self.play_board_canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=ghost_color,
                    outline=color,
                    width=2,
                    stipple='gray50'
                )
                self.ghost_items.append(item)

    def _clear_ghost(self):
        """Clear ghost preview."""
        for item in self.ghost_items:
            self.play_board_canvas.delete(item)
        self.ghost_items = []

    def _try_place_piece(self, row: int, col: int):
        """Try to place the selected piece at position."""
        if self.selected_piece_idx is None or not self.play_engine:
            return
        
        if not self.play_engine.can_place_piece(self.selected_piece_idx, row, col):
            self.play_status.config(text="Can't place there! Try another spot.")
            return
        
        # Get piece color before placing
        piece = self.play_engine.current_pieces[self.selected_piece_idx]
        piece_color = self.COLORS['piece_colors'][self.selected_piece_idx % len(self.COLORS['piece_colors'])]
        
        # Store colors for the piece blocks BEFORE making move
        for r, c in piece.blocks:
            cell_row = row + r
            cell_col = col + c
            if 0 <= cell_row < 8 and 0 <= cell_col < 8:
                self.board_colors[cell_row][cell_col] = piece_color
        
        # Make the move
        result = self.play_engine.make_move(self.selected_piece_idx, row, col)
        
        if result.success:
            # Flash effect for placement
            self._flash_placed_blocks(result)
            
            # Sync colors with board state - clear colors where board is now empty
            if result.lines_cleared > 0:
                board = self.play_engine.board.grid
                for r in range(8):
                    for c in range(8):
                        if board[r, c] == 0:
                            self.board_colors[r][c] = None
                
                combo_text = f"x{result.combo_multiplier}" if result.combo_multiplier > 1 else ""
                self.play_status.config(
                    text=f"Cleared {result.lines_cleared} line(s)! {combo_text} +{result.score_gained:,} pts"
                )
            else:
                self.play_status.config(text=f"+{result.score_gained} points")
            
            self._update_play_display()
            
            # Check game over
            if result.game_over:
                self._show_game_over()

    def _flash_placed_blocks(self, result):
        """Create a flash effect for placed blocks."""
        if not result.piece_placed or not result.position:
            return
        
        row, col = result.position
        piece = result.piece_placed
        
        # Create flash rectangles
        flash_items = []
        for r, c in piece.blocks:
            cell_row = row + r
            cell_col = col + c
            if 0 <= cell_row < 8 and 0 <= cell_col < 8:
                x1 = cell_col * self.play_cell_size + 1
                y1 = cell_row * self.play_cell_size + 1
                x2 = x1 + self.play_cell_size - 2
                y2 = y1 + self.play_cell_size - 2
                
                item = self.play_board_canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill='#ffffff',
                    outline='#ffffff',
                    width=0
                )
                flash_items.append(item)
        
        # Remove flash after delay
        self.root.after(100, lambda: self._remove_flash(flash_items))

    def _remove_flash(self, items):
        """Remove flash effect."""
        for item in items:
            self.play_board_canvas.delete(item)
        self._draw_play_board()

    def _show_game_over(self):
        """Show game over overlay."""
        self.final_score_label.config(text=f"Final Score: {self.play_engine.score:,}")
        self.game_over_frame.place(relx=0.5, rely=0.5, anchor='center')

    # ==================== TRAINING PAGE ====================

    def _create_training_page(self):
        """Create the training page with terminal output."""
        page = tk.Frame(self.container, bg=self.COLORS['bg'])
        self.pages['training'] = page

        # Header
        header = tk.Frame(page, bg=self.COLORS['panel'], height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        back_btn = tk.Button(
            header,
            text="< Back",
            font=('Segoe UI', 11),
            bg=self.COLORS['accent2'],
            fg=self.COLORS['text'],
            activebackground='#1a5a9a',
            relief='flat',
            cursor='hand2',
            command=self._back_from_training
        )
        back_btn.pack(side=tk.LEFT, padx=15, pady=12)

        title = tk.Label(
            header,
            text="Training Mode",
            font=('Segoe UI', 18, 'bold'),
            bg=self.COLORS['panel'],
            fg=self.COLORS['text']
        )
        title.pack(side=tk.LEFT, padx=20)

        # Main content
        content = tk.Frame(page, bg=self.COLORS['bg'])
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Left side - Controls
        left_panel = tk.Frame(content, bg=self.COLORS['panel'], width=280)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        left_panel.pack_propagate(False)

        controls_inner = tk.Frame(left_panel, bg=self.COLORS['panel'])
        controls_inner.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Control buttons
        ctrl_label = tk.Label(
            controls_inner,
            text="Training Controls",
            font=('Segoe UI', 14, 'bold'),
            bg=self.COLORS['panel'],
            fg=self.COLORS['text']
        )
        ctrl_label.pack(pady=(0, 20))

        self.train_btn = tk.Button(
            controls_inner,
            text="Start Training",
            font=('Segoe UI', 12, 'bold'),
            bg=self.COLORS['accent'],
            fg=self.COLORS['text'],
            activebackground='#ff6b6b',
            width=18,
            height=2,
            cursor='hand2',
            relief='flat',
            command=self._toggle_training
        )
        self.train_btn.pack(pady=10)

        self.clear_btn = tk.Button(
            controls_inner,
            text="Clear Output",
            font=('Segoe UI', 10),
            bg=self.COLORS['accent2'],
            fg=self.COLORS['text'],
            width=18,
            cursor='hand2',
            relief='flat',
            command=self._clear_terminal
        )
        self.clear_btn.pack(pady=5)

        # Stats section
        stats_label = tk.Label(
            controls_inner,
            text="Live Statistics",
            font=('Segoe UI', 12, 'bold'),
            bg=self.COLORS['panel'],
            fg=self.COLORS['text']
        )
        stats_label.pack(pady=(30, 15))

        self.train_stat_labels = {}
        stats_items = [
            ('steps', 'Steps'),
            ('episodes', 'Episodes'),
            ('avg_score', 'Avg Score'),
            ('best_score', 'Best Score'),
            ('fps', 'FPS'),
        ]

        for key, label_text in stats_items:
            frame = tk.Frame(controls_inner, bg=self.COLORS['panel'])
            frame.pack(fill=tk.X, pady=3)

            label = tk.Label(
                frame,
                text=f"{label_text}:",
                font=('Segoe UI', 10),
                bg=self.COLORS['panel'],
                fg=self.COLORS['text_dim'],
                anchor='w'
            )
            label.pack(side=tk.LEFT)

            value = tk.Label(
                frame,
                text="0",
                font=('Segoe UI', 11, 'bold'),
                bg=self.COLORS['panel'],
                fg=self.COLORS['accent'],
                anchor='e'
            )
            value.pack(side=tk.RIGHT)
            self.train_stat_labels[key] = value

        # Status
        self.train_status = tk.Label(
            controls_inner,
            text="Ready to train",
            font=('Segoe UI', 10),
            bg=self.COLORS['panel'],
            fg=self.COLORS['text_dim']
        )
        self.train_status.pack(pady=(30, 0))

        # Right side - Terminal
        right_panel = tk.Frame(content, bg=self.COLORS['terminal_bg'])
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        terminal_header = tk.Frame(right_panel, bg='#161b22', height=35)
        terminal_header.pack(fill=tk.X)
        terminal_header.pack_propagate(False)

        terminal_title = tk.Label(
            terminal_header,
            text="  Training Output",
            font=('Consolas', 10),
            bg='#161b22',
            fg=self.COLORS['text_dim'],
            anchor='w'
        )
        terminal_title.pack(side=tk.LEFT, pady=8)

        self.terminal = scrolledtext.ScrolledText(
            right_panel,
            font=('Consolas', 10),
            bg=self.COLORS['terminal_bg'],
            fg=self.COLORS['terminal_fg'],
            insertbackground=self.COLORS['terminal_fg'],
            relief='flat',
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.terminal.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Configure terminal colors
        self.terminal.tag_configure('info', foreground='#58a6ff')
        self.terminal.tag_configure('success', foreground='#3fb950')
        self.terminal.tag_configure('warning', foreground='#d29922')
        self.terminal.tag_configure('error', foreground='#f85149')

    def _back_from_training(self):
        """Handle back button from training page."""
        if self.is_training:
            if messagebox.askyesno("Stop Training?", "Training is in progress. Stop and go back?"):
                self._stop_training()
                self._show_page('menu')
        else:
            self._show_page('menu')

    def _clear_terminal(self):
        """Clear terminal output."""
        self.terminal.config(state=tk.NORMAL)
        self.terminal.delete('1.0', tk.END)
        self.terminal.config(state=tk.DISABLED)

    def _append_terminal(self, text: str, tag: str = None):
        """Append text to terminal."""
        self.terminal.config(state=tk.NORMAL)
        if tag:
            self.terminal.insert(tk.END, text + '\n', tag)
        else:
            self.terminal.insert(tk.END, text + '\n')
        self.terminal.see(tk.END)
        self.terminal.config(state=tk.DISABLED)

    def _toggle_training(self):
        """Toggle training state."""
        if self.is_training:
            self._stop_training()
        else:
            self._start_training()

    def _start_training(self):
        """Start training."""
        self.is_training = True
        self.train_btn.config(text="Stop Training", bg='#c0392b')
        self.train_status.config(text="Training in progress...")
        self._append_terminal("Starting training...", 'info')

        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.training_thread.start()

    def _stop_training(self):
        """Stop training."""
        self.is_training = False
        self.train_btn.config(text="Start Training", bg=self.COLORS['accent'])
        self.train_status.config(text="Training stopped")
        self._append_terminal("Training stopped by user", 'warning')

    def _training_loop(self):
        """Background training loop with output capture."""
        try:
            # Redirect stdout to capture training output
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = OutputRedirector(self.message_queue)
            sys.stderr = OutputRedirector(self.message_queue, 'stderr')

            from scripts.train import train, load_config

            config_path = project_root / "config" / "default.yaml"
            if config_path.exists():
                config = load_config(str(config_path))
            else:
                config = self._get_default_config()

            def progress_callback(metrics):
                if not self.is_training:
                    return False
                self.message_queue.put(('training_update', metrics))
                return True

            train(config, progress_callback=progress_callback)

        except Exception as e:
            self.message_queue.put(('error', str(e)))
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            self.message_queue.put(('training_done', None))

    # ==================== WATCH PAGE ====================

    def _create_watch_page(self):
        """Create the watch AI page."""
        page = tk.Frame(self.container, bg=self.COLORS['bg'])
        self.pages['watch'] = page

        # Header
        header = tk.Frame(page, bg=self.COLORS['panel'], height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        back_btn = tk.Button(
            header,
            text="< Back",
            font=('Segoe UI', 11),
            bg=self.COLORS['accent2'],
            fg=self.COLORS['text'],
            activebackground='#1a5a9a',
            relief='flat',
            cursor='hand2',
            command=self._back_from_watch
        )
        back_btn.pack(side=tk.LEFT, padx=15, pady=12)

        title = tk.Label(
            header,
            text="Watch AI Play",
            font=('Segoe UI', 18, 'bold'),
            bg=self.COLORS['panel'],
            fg=self.COLORS['text']
        )
        title.pack(side=tk.LEFT, padx=20)

        # Main content
        content = tk.Frame(page, bg=self.COLORS['bg'])
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Left side - Game board
        left_panel = tk.Frame(content, bg=self.COLORS['panel'])
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20))

        board_container = tk.Frame(left_panel, bg=self.COLORS['panel'])
        board_container.pack(expand=True)

        self.cell_size = 50
        board_size = self.cell_size * 8 + 2

        self.board_canvas = tk.Canvas(
            board_container,
            width=board_size,
            height=board_size,
            bg=self.COLORS['grid_bg'],
            highlightthickness=3,
            highlightbackground=self.COLORS['accent']
        )
        self.board_canvas.pack(pady=20)
        self._draw_board()

        # Pieces preview
        pieces_label = tk.Label(
            board_container,
            text="Current Pieces",
            font=('Segoe UI', 12, 'bold'),
            bg=self.COLORS['panel'],
            fg=self.COLORS['text']
        )
        pieces_label.pack(pady=(10, 5))

        pieces_frame = tk.Frame(board_container, bg=self.COLORS['panel'])
        pieces_frame.pack(pady=10)

        self.piece_canvases = []
        for i in range(3):
            canvas = tk.Canvas(
                pieces_frame,
                width=90,
                height=90,
                bg=self.COLORS['grid_bg'],
                highlightthickness=2,
                highlightbackground=self.COLORS['accent2']
            )
            canvas.pack(side=tk.LEFT, padx=8)
            self.piece_canvases.append(canvas)

        # Right side - Controls & Stats
        right_panel = tk.Frame(content, bg=self.COLORS['panel'], width=280)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel.pack_propagate(False)

        controls_inner = tk.Frame(right_panel, bg=self.COLORS['panel'])
        controls_inner.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Play controls
        ctrl_label = tk.Label(
            controls_inner,
            text="Playback Controls",
            font=('Segoe UI', 14, 'bold'),
            bg=self.COLORS['panel'],
            fg=self.COLORS['text']
        )
        ctrl_label.pack(pady=(0, 20))

        self.watch_btn = tk.Button(
            controls_inner,
            text="Start Watching",
            font=('Segoe UI', 12, 'bold'),
            bg=self.COLORS['accent'],
            fg=self.COLORS['text'],
            activebackground='#ff6b6b',
            width=18,
            height=2,
            cursor='hand2',
            relief='flat',
            command=self._toggle_watching
        )
        self.watch_btn.pack(pady=10)

        # Model selection
        model_label = tk.Label(
            controls_inner,
            text="Select Model",
            font=('Segoe UI', 10),
            bg=self.COLORS['panel'],
            fg=self.COLORS['text_dim']
        )
        model_label.pack(pady=(20, 5))

        self.model_var = tk.StringVar(value="best.pt")
        self.model_combo = ttk.Combobox(
            controls_inner,
            textvariable=self.model_var,
            values=self._get_available_models(),
            width=22,
            state='readonly'
        )
        self.model_combo.pack()

        # Speed control
        speed_label = tk.Label(
            controls_inner,
            text="Play Speed",
            font=('Segoe UI', 10),
            bg=self.COLORS['panel'],
            fg=self.COLORS['text_dim']
        )
        speed_label.pack(pady=(20, 5))

        self.speed_var = tk.IntVar(value=500)
        speed_scale = tk.Scale(
            controls_inner,
            from_=50,
            to=1000,
            orient=tk.HORIZONTAL,
            variable=self.speed_var,
            bg=self.COLORS['panel'],
            fg=self.COLORS['text'],
            highlightthickness=0,
            troughcolor=self.COLORS['accent2'],
            length=180,
            command=lambda v: setattr(self, 'watch_speed', int(v))
        )
        speed_scale.pack()

        # Stats
        stats_label = tk.Label(
            controls_inner,
            text="Game Statistics",
            font=('Segoe UI', 12, 'bold'),
            bg=self.COLORS['panel'],
            fg=self.COLORS['text']
        )
        stats_label.pack(pady=(30, 15))

        self.watch_stat_labels = {}
        stats_items = [
            ('score', 'Score'),
            ('best_score', 'Best Score'),
            ('moves', 'Moves'),
            ('lines', 'Lines'),
            ('combo', 'Combo'),
            ('games', 'Games'),
        ]

        for key, label_text in stats_items:
            frame = tk.Frame(controls_inner, bg=self.COLORS['panel'])
            frame.pack(fill=tk.X, pady=3)

            label = tk.Label(
                frame,
                text=f"{label_text}:",
                font=('Segoe UI', 10),
                bg=self.COLORS['panel'],
                fg=self.COLORS['text_dim'],
                anchor='w'
            )
            label.pack(side=tk.LEFT)

            value = tk.Label(
                frame,
                text="0",
                font=('Segoe UI', 11, 'bold'),
                bg=self.COLORS['panel'],
                fg=self.COLORS['accent'],
                anchor='e'
            )
            value.pack(side=tk.RIGHT)
            self.watch_stat_labels[key] = value

        # Status
        self.watch_status = tk.Label(
            controls_inner,
            text="Select a model and click Start",
            font=('Segoe UI', 10),
            bg=self.COLORS['panel'],
            fg=self.COLORS['text_dim']
        )
        self.watch_status.pack(pady=(20, 0))

    def _back_from_watch(self):
        """Handle back button from watch page."""
        if self.is_watching:
            self._stop_watching()
        self._show_page('menu')

    def _draw_board(self):
        """Draw the game board."""
        self.board_canvas.delete('all')

        for row in range(8):
            for col in range(8):
                x1 = col * self.cell_size + 1
                y1 = row * self.cell_size + 1
                x2 = x1 + self.cell_size - 2
                y2 = y1 + self.cell_size - 2

                if self.board_state[row, col]:
                    # Use tracked color if available
                    color = None
                    if hasattr(self, 'watch_board_colors') and self.watch_board_colors:
                        color = self.watch_board_colors[row][col]
                    if not color:
                        color = self.COLORS['piece_colors'][0]
                    self._draw_3d_block(self.board_canvas, x1, y1, x2, y2, color)
                else:
                    self.board_canvas.create_rectangle(
                        x1, y1, x2, y2,
                        fill=self.COLORS['empty_cell'],
                        outline='#2d3748',
                        width=1
                    )

    def _draw_pieces(self, pieces: List[np.ndarray], pieces_used: List[bool] = None):
        """Draw piece previews with 3D style."""
        if pieces_used is None:
            pieces_used = [False, False, False]
        
        for i, canvas in enumerate(self.piece_canvases):
            canvas.delete('all')

            # Check if piece is used
            if i < len(pieces_used) and pieces_used[i]:
                # Show checkmark for used piece
                canvas.create_text(
                    45, 45,
                    text="✓",
                    font=('Segoe UI', 24),
                    fill=self.COLORS['text_dim']
                )
            elif i < len(pieces) and pieces[i] is not None:
                piece = pieces[i]
                cell_size = 18

                h, w = piece.shape
                offset_x = (90 - w * cell_size) // 2
                offset_y = (90 - h * cell_size) // 2

                color = self.COLORS['piece_colors'][i % len(self.COLORS['piece_colors'])]

                for row in range(h):
                    for col in range(w):
                        if piece[row, col]:
                            x1 = offset_x + col * cell_size
                            y1 = offset_y + row * cell_size
                            x2 = x1 + cell_size - 2
                            y2 = y1 + cell_size - 2

                            self._draw_3d_block(canvas, x1, y1, x2, y2, color)

    def _lighten_color(self, hex_color: str, amount: float = 0.2) -> str:
        """Lighten a hex color."""
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        new_rgb = tuple(min(255, int(c + (255 - c) * amount)) for c in rgb)
        return f'#{new_rgb[0]:02x}{new_rgb[1]:02x}{new_rgb[2]:02x}'

    def _darken_color(self, hex_color: str, amount: float = 0.3) -> str:
        """Darken a hex color."""
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        new_rgb = tuple(max(0, int(c * (1 - amount))) for c in rgb)
        return f'#{new_rgb[0]:02x}{new_rgb[1]:02x}{new_rgb[2]:02x}'

    def _get_available_models(self) -> List[str]:
        """Get available model checkpoints."""
        checkpoint_dir = project_root / "checkpoints"
        models = ["best.pt", "latest.pt", "final.pt"]

        if checkpoint_dir.exists():
            for f in checkpoint_dir.glob("*.pt"):
                if f.name not in models:
                    models.append(f.name)

        return models

    def _toggle_watching(self):
        """Toggle watch state."""
        if self.is_watching:
            self._stop_watching()
        else:
            self._start_watching()

    def _start_watching(self):
        """Start watching AI play."""
        model_name = self.model_var.get()
        model_path = project_root / "checkpoints" / model_name

        if not model_path.exists():
            messagebox.showerror("Error", f"Model not found: {model_path}")
            return

        # Reset color tracking for new watch session
        self.watch_board_colors = [[None for _ in range(8)] for _ in range(8)]
        self.watch_piece_color_idx = 0

        self.is_watching = True
        self.watch_btn.config(text="Stop Watching", bg='#c0392b')
        self.watch_status.config(text="AI is playing...")

        self.watch_thread = threading.Thread(
            target=self._watch_loop,
            args=(str(model_path),),
            daemon=True
        )
        self.watch_thread.start()

    def _stop_watching(self):
        """Stop watching."""
        self.is_watching = False
        self.watch_btn.config(text="Start Watching", bg=self.COLORS['accent'])
        self.watch_status.config(text="Stopped")

    def _watch_loop(self, model_path: str):
        """Background watch loop."""
        try:
            # Ensure paths are available in this thread
            import sys
            from pathlib import Path
            thread_project_root = Path(__file__).parent.parent
            sys.path.insert(0, str(thread_project_root))
            sys.path.insert(0, str(thread_project_root / "src"))
            
            import torch
            from game.engine import GameEngine
            from models.network import BlockBlastNetwork
            from environment.block_blast_env import BlockBlastEnv

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            network = BlockBlastNetwork().to(device)

            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            if 'network_state_dict' in checkpoint:
                network.load_state_dict(checkpoint['network_state_dict'])
            elif 'model_state_dict' in checkpoint:
                network.load_state_dict(checkpoint['model_state_dict'])
            else:
                network.load_state_dict(checkpoint)
            network.eval()

            games_played = 0
            best_score = 0

            while self.is_watching:
                env = BlockBlastEnv()
                obs, info = env.reset()

                total_score = 0
                moves = 0

                while self.is_watching:
                    with torch.no_grad():
                        board = torch.FloatTensor(obs['board']).unsqueeze(0).to(device)
                        pieces = torch.FloatTensor(obs['pieces']).unsqueeze(0).to(device)
                        mask = torch.FloatTensor(obs['action_mask']).unsqueeze(0).to(device)

                        action, _, _, _ = network.get_action_and_value(
                            board, pieces, mask, deterministic=True
                        )
                        action = action.item()

                    obs, reward, terminated, truncated, info = env.step(action)

                    total_score = info.get('score', total_score)
                    moves += 1
                    lines = info.get('lines_cleared_total', 0)
                    combo = info.get('combo', 0)

                    self.message_queue.put(('game_state', {
                        'board': env.engine.board.grid.copy(),
                        'pieces': [p.get_shape_array() if p else None for p in env.engine.current_pieces],
                        'pieces_used': env.engine.pieces_used.copy(),
                        'score': total_score,
                        'moves': moves,
                        'lines': lines,
                        'combo': combo,
                        'games': games_played,
                        'best_score': best_score,
                    }))

                    time.sleep(self.watch_speed / 1000.0)

                    if terminated or truncated:
                        games_played += 1
                        best_score = max(best_score, total_score)
                        time.sleep(1)
                        break

                env.close()

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            self.message_queue.put(('error', error_msg))
        finally:
            self.message_queue.put(('watch_done', None))

    # ==================== MESSAGE QUEUE ====================

    def _check_queue(self):
        """Process message queue."""
        try:
            while True:
                msg_type, data = self.message_queue.get_nowait()

                if msg_type == 'output':
                    self._append_terminal(data)
                elif msg_type == 'game_state':
                    self._update_game_state(data)
                elif msg_type == 'training_update':
                    self._update_training_stats(data)
                elif msg_type == 'training_done':
                    if self.is_training:
                        self._stop_training()
                elif msg_type == 'watch_done':
                    if self.is_watching:
                        self._stop_watching()
                elif msg_type == 'error':
                    error_text = str(data)
                    # Show first 500 chars in messagebox, full error in terminal
                    short_error = error_text[:500] + ("..." if len(error_text) > 500 else "")
                    messagebox.showerror("Error", short_error)
                    # Try to append to terminal if we're on training page
                    if hasattr(self, 'terminal'):
                        self._append_terminal(f"ERROR: {error_text}", 'error')
                    # Update watch status if we're on watch page
                    if hasattr(self, 'watch_status') and self.is_watching:
                        self.watch_status.config(text="Error occurred - check message box")
                        self._stop_watching()
        except queue.Empty:
            pass

        self.root.after(50, self._check_queue)

    def _update_game_state(self, data: Dict[str, Any]):
        """Update game visualization."""
        if 'board' in data:
            new_board = data['board']
            
            # Initialize watch color tracking if needed
            if not hasattr(self, 'watch_board_colors') or self.watch_board_colors is None:
                self.watch_board_colors = [[None for _ in range(8)] for _ in range(8)]
                self.watch_piece_color_idx = 0
            
            # Detect newly placed blocks and assign same color
            new_blocks = []
            for r in range(8):
                for c in range(8):
                    # New block appeared
                    if new_board[r, c] and not self.board_state[r, c]:
                        new_blocks.append((r, c))
                    # Block disappeared (line clear)
                    elif not new_board[r, c] and self.board_state[r, c]:
                        self.watch_board_colors[r][c] = None
            
            # All new blocks get the same color (same piece)
            if new_blocks:
                color = self.COLORS['piece_colors'][self.watch_piece_color_idx % len(self.COLORS['piece_colors'])]
                for r, c in new_blocks:
                    self.watch_board_colors[r][c] = color
                self.watch_piece_color_idx += 1
            
            self.board_state = new_board
            self._draw_board()

        if 'pieces' in data:
            pieces_used = data.get('pieces_used', [False, False, False])
            self._draw_pieces(data['pieces'], pieces_used)

        for key in ['score', 'moves', 'lines', 'combo', 'games', 'best_score']:
            if key in data and key in self.watch_stat_labels:
                self.watch_stat_labels[key].config(text=f"{data[key]:,}")

    def _update_training_stats(self, metrics: Dict[str, Any]):
        """Update training statistics."""
        mapping = {
            'total_steps': 'steps',
            'mean_score': 'avg_score',
            'best_score': 'best_score',
            'episodes': 'episodes',
            'fps': 'fps',
        }

        for src, dst in mapping.items():
            if src in metrics and dst in self.train_stat_labels:
                value = metrics[src]
                if isinstance(value, float):
                    text = f"{value:,.1f}"
                else:
                    text = f"{value:,}"
                self.train_stat_labels[dst].config(text=text)

    def _get_default_config(self) -> Dict:
        """Get default training config from config/default.yaml."""
        config_path = project_root / "config" / "default.yaml"
        
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    return config
        except Exception as e:
            print(f"Warning: Could not load {config_path}: {e}")
        
        # Fallback to hardcoded config if file doesn't exist or fails to load
        return {
            'environment': {'board_size': 8},
            'ppo': {
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_epsilon': 0.2,
                'entropy_coef': 0.01,
                'value_coef': 0.5,
                'max_grad_norm': 0.5,
            },
            'training': {
                'num_envs': 64,
                'batch_size': 2048,
                'num_epochs': 10,
                'rollout_steps': 128,
                'total_timesteps': 10_000_000,
            },
            'rewards': {
                'line_clear_base': 1.0,
                'block_placed': 0.01,
                'game_over_penalty': -1.0,
            },
            'logging': {
                'log_interval': 10,
                'save_interval': 100,
                'eval_interval': 50,
            },
            'paths': {
                'checkpoint_dir': 'checkpoints',
                'log_dir': 'logs',
                'results_dir': 'results',
            },
        }


def main():
    """Launch the GUI."""
    root = tk.Tk()

    icon_path = project_root / "gui" / "icon.ico"
    if icon_path.exists():
        root.iconbitmap(str(icon_path))

    app = BlockBlastGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
