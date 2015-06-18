import os
import re
import subprocess
import shutil


class NbConverter:
    """Jupyter Notebook Convert to ReST"""

    def __init__(self, top='./', ext='rst', tab='    '):
        self.top = top
        self.ext = ext
        self.notebooks = []
        self.tab = tab

        self.cwd = os.getcwd()
        self.src_pattern = re.compile('^(\.\. +\w+:: )(.+_files)')

    def search(self, exclude=[]):

        if self.notebooks:
            return self.notebooks

        for root, dirs, files in os.walk(self.top, topdown=True):

            dirs[:] = [d for d in dirs if d not in exclude]
            self.notebooks.extend((root, filename) for filename in files
                            if filename.endswith('.ipynb')
                            and not filename.endswith('-checkpoint.ipynb'))
        return self.notebooks

    def convert_all(self, exclude=[]):

        for root, nb in self.search(exclude):
            self.convert(root, nb)
        os.chdir(self.cwd)

    def convert(self, root, nb, statdir='_static'):

        root = root.replace('./','')
        os.chdir(os.path.join(self.cwd, root))

        print("**********************************************")
        print("[NotebookConvert] Converting: {}".format(nb))
        print("**********************************************")

        try:
            subprocess.check_call(['ipython', 'nbconvert', nb, '--to', self.ext])
        except subprocess.CalledProcessError:
            print("[NotebookConvert] FAIL: {}".format(os.path.join(root, nb)))
            return
        else:
            print("[NotebookConvert] SUCCESS: {}".format(os.path.join(root, nb)))


        statdir_rel = os.path.join(statdir, root,  nb[:-6] + '_files')
        statdir_abs = os.path.join(self.cwd, statdir_rel)

        try:
            shutil.rmtree(statdir_abs)
        except OSError:
            pass

        try:
            shutil.move(nb[:-6] + '_files', statdir_abs)
        except IOError:
            pass

        with open(nb[:-6] + '.rst', 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            lines[i] = self.src_pattern.sub(r"\1/_static/" + root + r"/\2", line)


        n, i = len(lines), 0
        rawWatch = False

        while i < n:
            line = lines[i]
            if line.startswith('.. image::'):
                lines.insert(i + 1, self.tab + ':class: pynb\n')
                n += 1

            elif line.startswith('.. parsed-literal::'):
                lines.insert(i + 1, self.tab + ':class: pynb-result\n')
                n += 1

            elif line == '::\n':
                del lines[i]
                n -= 1
                i -= 1


            elif line.startswith('.. raw:: html'):
                rawWatch = True

            if rawWatch:
                if '<div' in line:
                    line = line.replace('<div', '<div class="pynb-result"')
                    lines[i] = line
                    rawWatch = False
            i += 1

        with open(nb[:-6] + '.rst', 'w') as f:
            f.writelines(lines)


if __name__ == "__main__":

    nbconvert = NbConverter(top='./', ext='rst')
    nbconvert.convert_all(exclude=['.git', '_build', '_static',
                                    '_template', '.ipynb_checkpoints'])
