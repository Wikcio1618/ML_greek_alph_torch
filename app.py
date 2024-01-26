import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import torch
import os
from PIL import Image, ImageDraw, ImageTk 
from model import GreekLettersCNN

class DrawingBoard:

	path_to_model = 'model.pth'
	alphabet = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta',
				'eta', 'theta', 'iota', 'kappa', 'lambda', 'miu', 'niu',
				'ksi', 'omicron', 'pi', 'rho', 'sigma', 'tau', 'ypsilon',
				'phi', 'chi', 'psi', 'omega']

	def __init__(self):
		if os.path.isfile(self.path_to_model):
			self.model = torch.load(self.path_to_model)

			self.root = tk.Tk()
			self.root.title("Greek Letter Prediction")
			self.root.geometry("500x550")  # Slightly larger window size

			# Set a blue theme
			self.root.configure(bg='#87CEEB')

			self.canvas = tk.Canvas(self.root, width=450, height=450, bg='white', borderwidth=5, relief="ridge")
			self.canvas.pack()

			self.image = Image.new("L", (450, 450), color=255)
			self.draw = ImageDraw.Draw(self.image)

			self.prediction_frame = tk.Frame(self.root, bg='#ADD8E6')  # Bar at the bottom
			self.prediction_frame.pack(fill=tk.X, pady=(0, 10))

			self.prediction_label = tk.Label(self.prediction_frame, text="Model Prediction: ", font=('Helvetica', 14, 'bold'), bg='#ADD8E6')
			self.prediction_label.pack(side=tk.BOTTOM, padx=10)

			self.clear_button = tk.Button(self.root, text="Clear Drawing", command=self.clear_drawing, font=('Helvetica', 12), bg='#ADD8E6')
			self.clear_button.pack(side=tk.BOTTOM, padx=10, pady=10)
	
			self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
			self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)

	def on_mouse_drag(self, event):
		x, y = event.x, event.y
		radius = 16
		self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill='black')
		self.draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=0)

	def display_resized_image(self, image):
		# Convert the NumPy array to a PhotoImage
		tk_image = self.numpy_to_photoimage(image)

		# Display the resized image on the right canvas
		self.resized_canvas.create_image(0, 0, anchor = tk.NW, image=tk_image)

	def on_mouse_release(self, event):
		image = self.get_canvas_as_array()
		image = image.reshape(1, 1, 14, 14)
		image = torch.tensor(image, dtype=torch.float32)
		with torch.no_grad():
			self.model.eval()
			output = self.model(image)
			guess = torch.argmax(output, dim=1)

		self.prediction_label.config(text=f"Model Prediction: {self.alphabet[guess]}")

		self.root.update()
		

	def clear_drawing(self):
		self.canvas.delete("all")
		self.image = Image.new("L", (450, 450), color=255)
		self.draw = ImageDraw.Draw(self.image)
		self.prediction_label.config(text="Model Prediction: ")

	def get_canvas_as_array(self):
		resized_image = self.image.resize((14, 14))
		canvas_array = np.array(resized_image)
		return canvas_array

	def run(self):
		self.root.mainloop()

if __name__ == "__main__":
	drawing_board = DrawingBoard()
	drawing_board.run()
