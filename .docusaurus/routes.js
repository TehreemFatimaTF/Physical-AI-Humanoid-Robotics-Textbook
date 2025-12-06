import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/physical-ai-robotics-textbook/__docusaurus/debug',
    component: ComponentCreator('/physical-ai-robotics-textbook/__docusaurus/debug', '51c'),
    exact: true
  },
  {
    path: '/physical-ai-robotics-textbook/__docusaurus/debug/config',
    component: ComponentCreator('/physical-ai-robotics-textbook/__docusaurus/debug/config', 'c7a'),
    exact: true
  },
  {
    path: '/physical-ai-robotics-textbook/__docusaurus/debug/content',
    component: ComponentCreator('/physical-ai-robotics-textbook/__docusaurus/debug/content', '6d1'),
    exact: true
  },
  {
    path: '/physical-ai-robotics-textbook/__docusaurus/debug/globalData',
    component: ComponentCreator('/physical-ai-robotics-textbook/__docusaurus/debug/globalData', '71d'),
    exact: true
  },
  {
    path: '/physical-ai-robotics-textbook/__docusaurus/debug/metadata',
    component: ComponentCreator('/physical-ai-robotics-textbook/__docusaurus/debug/metadata', '9b0'),
    exact: true
  },
  {
    path: '/physical-ai-robotics-textbook/__docusaurus/debug/registry',
    component: ComponentCreator('/physical-ai-robotics-textbook/__docusaurus/debug/registry', '8fa'),
    exact: true
  },
  {
    path: '/physical-ai-robotics-textbook/__docusaurus/debug/routes',
    component: ComponentCreator('/physical-ai-robotics-textbook/__docusaurus/debug/routes', 'eb6'),
    exact: true
  },
  {
    path: '/physical-ai-robotics-textbook/modules',
    component: ComponentCreator('/physical-ai-robotics-textbook/modules', 'c96'),
    exact: true
  },
  {
    path: '/physical-ai-robotics-textbook/docs',
    component: ComponentCreator('/physical-ai-robotics-textbook/docs', 'adb'),
    routes: [
      {
        path: '/physical-ai-robotics-textbook/docs/advanced-ai-control/module-5-advanced-ai',
        component: ComponentCreator('/physical-ai-robotics-textbook/docs/advanced-ai-control/module-5-advanced-ai', '168'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/physical-ai-robotics-textbook/docs/appendix/glossary',
        component: ComponentCreator('/physical-ai-robotics-textbook/docs/appendix/glossary', '166'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/physical-ai-robotics-textbook/docs/appendix/references',
        component: ComponentCreator('/physical-ai-robotics-textbook/docs/appendix/references', '6e6'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/physical-ai-robotics-textbook/docs/appendix/resources',
        component: ComponentCreator('/physical-ai-robotics-textbook/docs/appendix/resources', '5fe'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/physical-ai-robotics-textbook/docs/hardware-basics/module-3-hardware',
        component: ComponentCreator('/physical-ai-robotics-textbook/docs/hardware-basics/module-3-hardware', '243'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/physical-ai-robotics-textbook/docs/humanoid-design/module-6-humanoid-design',
        component: ComponentCreator('/physical-ai-robotics-textbook/docs/humanoid-design/module-6-humanoid-design', '6b0'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/physical-ai-robotics-textbook/docs/intro',
        component: ComponentCreator('/physical-ai-robotics-textbook/docs/intro', '3e7'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/physical-ai-robotics-textbook/docs/introduction/intro',
        component: ComponentCreator('/physical-ai-robotics-textbook/docs/introduction/intro', 'bcc'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/physical-ai-robotics-textbook/docs/ros2-foundations/module-1-ros2',
        component: ComponentCreator('/physical-ai-robotics-textbook/docs/ros2-foundations/module-1-ros2', '60e'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/physical-ai-robotics-textbook/docs/ros2-foundations/ros2-hands-on',
        component: ComponentCreator('/physical-ai-robotics-textbook/docs/ros2-foundations/ros2-hands-on', '0e4'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/physical-ai-robotics-textbook/docs/simulation/digital-twins',
        component: ComponentCreator('/physical-ai-robotics-textbook/docs/simulation/digital-twins', '4f4'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/physical-ai-robotics-textbook/docs/simulation/gazebo-unity',
        component: ComponentCreator('/physical-ai-robotics-textbook/docs/simulation/gazebo-unity', 'eee'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/physical-ai-robotics-textbook/docs/simulation/module-2-simulation',
        component: ComponentCreator('/physical-ai-robotics-textbook/docs/simulation/module-2-simulation', '7cc'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/physical-ai-robotics-textbook/docs/vla-systems/module-4-vla-foundations',
        component: ComponentCreator('/physical-ai-robotics-textbook/docs/vla-systems/module-4-vla-foundations', '335'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/physical-ai-robotics-textbook/docs/vla-systems/vla-action',
        component: ComponentCreator('/physical-ai-robotics-textbook/docs/vla-systems/vla-action', 'f3f'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/physical-ai-robotics-textbook/docs/vla-systems/vla-hands-on-basic',
        component: ComponentCreator('/physical-ai-robotics-textbook/docs/vla-systems/vla-hands-on-basic', '709'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/physical-ai-robotics-textbook/docs/vla-systems/vla-language',
        component: ComponentCreator('/physical-ai-robotics-textbook/docs/vla-systems/vla-language', '716'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/physical-ai-robotics-textbook/docs/vla-systems/vla-vision',
        component: ComponentCreator('/physical-ai-robotics-textbook/docs/vla-systems/vla-vision', '980'),
        exact: true,
        sidebar: "tutorialSidebar"
      }
    ]
  },
  {
    path: '/physical-ai-robotics-textbook/',
    component: ComponentCreator('/physical-ai-robotics-textbook/', 'fec'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
