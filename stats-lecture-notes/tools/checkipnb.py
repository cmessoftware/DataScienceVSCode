#!/usr/bin/env python
"""
simple example script for running notebooks and reporting exceptions.

Usage: `checkipnb.py foo.ipynb [bar.ipynb [...]]`

Each cell is submitted to the kernel, and checked for errors.

Thanks MinRK:

https://gist.github.com/minrk/2620876#file-checkipnb-py
"""

import os,sys,time

from Queue import Empty

try:
    from IPython.kernel import KernelManager
except ImportError:
    from IPython.zmq.blockingkernelmanager import BlockingKernelManager as KernelManager

from IPython.nbformat.current import reads, NotebookNode

def run_notebook(nb):
    km = KernelManager()
    km.start_kernel(extra_arguments=['--profile', 'stats'])#, stderr=open(os.devnull, 'w'))
    try:
        kc = km.client()
    except AttributeError:
        # 0.13
        kc = km
    kc.start_channels()
    shell = kc.shell_channel
    # simple ping:
    shell.execute("pass")
    shell.get_msg()
    
    shell.execute("datadir = '%s'" % os.path.abspath(os.path.join( \
                os.path.abspath(os.path.dirname(__file__)), '..', 'data')))

    cells = 0
    failures = 0
    for ws in nb.worksheets:
        for cell in ws.cells:
            if cell.cell_type != 'code':
                continue
            shell.execute(cell.input)
            # wait for finish, maximum 20s
            msg = shell.get_msg(timeout=20)

            reply = msg['content']
            if reply['status'] == 'error':
                failures += 1
                print "\nFAILURE:"
                print cell.input
                print '-----'
                print "raised:"
                print '\n'.join(reply['traceback'])
            cells += 1
            sys.stdout.write('.')

    print "ran notebook %s" % nb.metadata.name
    print "    ran %3i cells" % cells
    if failures:
        print "    %3i cells raised exceptions" % failures
    kc.stop_channels()
    km.shutdown_kernel()
    del km

if __name__ == '__main__':
    for ipynb in sys.argv[1:]:
        print "running %s" % ipynb
        with open(ipynb) as f:
            nb = reads(f.read(), 'json')
        run_notebook(nb)
