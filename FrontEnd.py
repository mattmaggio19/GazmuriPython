from tkinter import Tk, Label, Button, Entry
import tkinter.filedialog as tkFiledialog



class MyFirstGUI:
    def __init__(self, master):
        self.master = master
        master.title("A simple GUI")

        self.label = Label(master, text="This is our first GUI!")
        self.label.pack()

        # self.FileViewer = tkFiledialog.askopenfilename()
        # self.FileViewer.pack()

        self.entry = Button(master, text = 'Master Data sheet path', command=self.FindMasterFile)
        self.entry.pack()
        #
        # self.greet_button = Button(master, text="Greet", command=self.greet)
        # self.greet_button.pack()

        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.pack()

    def FindMasterFile(self):
        File = tkFiledialog.askopenfilename()

        print(File)
    def greet(self):
        print("Greetings!")

root = Tk()
my_gui = MyFirstGUI(root)
root.mainloop()