#!/usr/bin/python
from scipy.io import loadmat
import os, sys, argparse

TREE_SYMBOL = '----'
SHIFT_SYMBOL = '    '

def lsmat(fn, compact_output = False):
    try:
        f = loadmat(fn, struct_as_record = True)
    except IOError:
        print "File does not exist"
        return 0
    goodkeys = f.keys()
    s = ''
    for key in goodkeys:
        s += "### %s ###\n" % key
        s += shift_str(_print_tree(f[key], compact_output = compact_output)) + '\n\n'
    print s

def shift_str(s):
    if s is None:
        return ''
    if not s[0] == '\n':
        s = '\n'+s
        return shift_str(s)
    else:
        return s.replace('\n', '\n    ')

def _print_tree(node, compact_output):
    if isinstance(node, str):
        return node
    elif isinstance(node, list):
        return 'List '+str(node)+'\n'
    elif node.shape == (1,):
        return _print_tree(node[0], compact_output)
    elif node.shape == (1,1):
        return _print_tree(node[0, 0], compact_output)
    elif node.shape == ():
        if node.dtype.names is None:
            out = ' * '+str(node)
            return shift_str(out)
        else:
            names = list(node.dtype.names)
            total = ''
            for name in names:
                out = shift_str(' * '+str(name))
                next = shift_str(_print_tree(node[name], compact_output=compact_output))
                total += out+next
            return total
    else:
        # if an array has to be printed out
        out = shift_str(TREE_SYMBOL+' * shape: ' + str(node.shape))
        if node.shape != (0,0) and not compact_output:
            out += shift_str(str(node))
        return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='List the contents of a MATLAB .mat file (version <7.3).')
    parser.add_argument('filename', metavar='filename', type=str, nargs=1,
                       help='path to the matfile')
    parser.add_argument('-c', help = 'Compact output (no array)', action = 'store_const',
                        dest = 'compact_output', const = True, default = False)
    args = parser.parse_args()
    fn = os.path.abspath(args.filename[0])
    print 'Printing Matlab file:\n     %s' % fn
    lsmat(fn, compact_output = args.compact_output)