"""
create_artifact_info.py

Pops up a small form asking for a Name, Type, and Description, then
writes an info.txt file formatted like:

    Name: <name>
    Type: <type>
    Description: <description>

Usage:
    python create_artifact_info.py
"""

import os
import tkinter as tk
from tkinter import messagebox
import textwrap


def wrap_description(text, width=70):
    """Wrap long description text to a fixed width, like the example file."""
    paragraphs = text.splitlines() or [""]
    wrapped_lines = []
    for para in paragraphs:
        if para.strip() == "":
            wrapped_lines.append("")
        else:
            wrapped_lines.extend(textwrap.wrap(para, width=width))
    return "\n".join(wrapped_lines)


def submit():
    name = name_entry.get().strip()
    type_ = type_var.get().strip()
    description = description_text.get("1.0", tk.END).strip()

    if not name or not type_ or not description:
        messagebox.showwarning(
            "Missing information",
            "Please fill in Name, Type, and Description before submitting."
        )
        return

    # Save directly into CAPTURE_DATA_DIR as info.txt
    save_dir = os.environ.get("CAPTURE_DATA_DIR")
    if not save_dir:
        messagebox.showerror(
            "CAPTURE_DATA_DIR not set",
            "The CAPTURE_DATA_DIR environment variable is not set.\n"
            "Please set it before running this script."
        )
        return

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "info.txt")

    content = (
        f"Name: {name}\n"
        f"Type: {type_}\n"
        f"Description: {wrap_description(description)}\n"
    )

    try:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(content)
    except OSError as e:
        messagebox.showerror("Error saving file", str(e))
        return

    messagebox.showinfo("Saved", f"Saved to:\n{save_path}")
    root.destroy()


# --- Build the popup window ---
root = tk.Tk()
root.title("New Artifact Info")
root.geometry("420x360")
root.resizable(False, False)

padding = {"padx": 12, "pady": 6}

tk.Label(root, text="Name:").pack(anchor="w", **padding)
name_entry = tk.Entry(root, width=45)
name_entry.pack(**padding)

tk.Label(root, text="Type:").pack(anchor="w", **padding)
type_var = tk.StringVar(root)
type_var.set("papyrus")  # default selection
type_dropdown = tk.OptionMenu(root, type_var, "papyrus", "tablet")
type_dropdown.config(width=20)
type_dropdown.pack(**padding)

tk.Label(root, text="Description:").pack(anchor="w", **padding)
description_text = tk.Text(root, width=45, height=8, wrap="word")
description_text.pack(**padding)

submit_button = tk.Button(root, text="Generate .txt", command=submit)
submit_button.pack(pady=12)

name_entry.focus_set()
root.mainloop()