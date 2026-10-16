# Using your Colab compute units from VS Code

Two established community techniques for this; neither is an official
Google feature, and Colab's Terms of Service explicitly call out remote-shell
use as against the rules for the **free** tier. Both are, in practice,
extremely widely used with **paid compute units** (Pro/Pro+, pay-as-you-go)
specifically for this purpose, but you're using your own paid units at your
own discretion either way.

## Option A -- `colab-ssh` + VS Code Remote-SSH (recommended: full remote dev)

Gives you a full VS Code remote window running ON the Colab VM: terminal,
file explorer, the works -- edit `Master_Experiments.ipynb` (or any file)
with the VM's GPU backing every cell you run.

1. In a Colab notebook cell (any notebook, doesn't need to be this one),
   using the GPU runtime you want to work on:
   ```python
   !pip install colab-ssh --upgrade
   from colab_ssh import launch_ssh_cloudflared
   launch_ssh_cloudflared(password="pick-a-password")
   ```
   This tunnels SSH out through Cloudflare (no port forwarding needed on your
   end) and prints an SSH config block.
2. Copy that block into your local `~/.ssh/config` (on Windows:
   `C:\Users\<you>\.ssh\config`).
3. In VS Code, install the **Remote - SSH** extension if you don't have it,
   then `Ctrl+Shift+P` -> "Remote-SSH: Connect to Host" -> pick the host
   `colab-ssh`/`colab` (whatever name the printed config used).
4. VS Code reopens as a remote window on the Colab VM. Install the
   **Jupyter** extension inside that remote window (extensions are
   per-connection), then open `Master_Experiments.ipynb` from wherever you
   put the `projects_clean` folder on that VM (see below) and run cells
   normally -- they execute on the Colab GPU.
5. The Colab session still has the usual idle/max-runtime limits; when it
   disconnects, redo steps 1-3 (a new tunnel each time).

Getting `vae_topology/` onto the VM for this option: either `git clone` your
repo inside the Colab terminal (VS Code's integrated terminal, now remote),
or `!cp -r /content/drive/MyDrive/.../projects_clean /content/projects_clean`
after mounting Drive from a cell.

## Option B -- ngrok + `jupyter_http_over_ws` (lighter-weight: kernel only)

No file browser or terminal, just a Jupyter kernel VS Code can attach to --
enough to run this notebook, not to browse the VM's filesystem from VS
Code's sidebar.

1. In a Colab cell:
   ```python
   !pip install jupyter_http_over_ws pyngrok
   !jupyter serverextension enable --py jupyter_http_over_ws

   get_ipython().system_raw(
       'jupyter notebook --NotebookApp.allow_origin="*" '
       '--ip 0.0.0.0 --port=8888 --NotebookApp.token="choose-a-token" &'
   )

   from pyngrok import ngrok
   ngrok.set_auth_token("YOUR_NGROK_AUTHTOKEN")   # free account at ngrok.com
   print(ngrok.connect(8888))
   ```
2. In VS Code (Jupyter extension installed), open `Master_Experiments.ipynb`,
   click the kernel picker (top right of the notebook) -> "Select Another
   Kernel" -> "Existing Jupyter Server" -> paste the `ngrok` URL, then the
   token you chose.
3. Cells now execute on the Colab GPU; the notebook file itself stays local
   (or wherever VS Code opened it from) -- only the kernel is remote.

Getting `vae_topology/` importable for this option: the KERNEL's filesystem
is the Colab VM, so the package still needs to physically exist there (Drive
mount + `sys.path.insert`, same as the notebook's own Section 1.2) --
opening the `.ipynb` locally in VS Code does not transfer local files to the
remote kernel.

## Either way: getting the code there

The notebook's Section 1.2 handles this at run time (mount Drive, `cd` into
wherever you put `projects_clean`). The one-time step is getting
`projects_clean/vae_topology` (and this notebook) onto Drive, or into a
GitHub repo you `git clone` from the Colab terminal -- pick whichever you
already use for the rest of the thesis's version control. The original
`Colab_run_files` notebooks used
`/content/drive/MyDrive/Bachelorarbeit/projects`; keeping a parallel
`/content/drive/MyDrive/Bachelorarbeit/projects_clean` is the least
surprising choice if you want both around during the transition.
